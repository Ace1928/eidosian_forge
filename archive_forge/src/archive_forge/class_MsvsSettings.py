import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
class MsvsSettings:
    """A class that understands the gyp 'msvs_...' values (especially the
    msvs_settings field). They largely correpond to the VS2008 IDE DOM. This
    class helps map those settings to command line options."""

    def __init__(self, spec, generator_flags):
        self.spec = spec
        self.vs_version = GetVSVersion(generator_flags)
        supported_fields = [('msvs_configuration_attributes', dict), ('msvs_settings', dict), ('msvs_system_include_dirs', list), ('msvs_disabled_warnings', list), ('msvs_precompiled_header', str), ('msvs_precompiled_source', str), ('msvs_configuration_platform', str), ('msvs_target_platform', str)]
        configs = spec['configurations']
        for field, default in supported_fields:
            setattr(self, field, {})
            for configname, config in configs.items():
                getattr(self, field)[configname] = config.get(field, default())
        self.msvs_cygwin_dirs = spec.get('msvs_cygwin_dirs', ['.'])
        unsupported_fields = ['msvs_prebuild', 'msvs_postbuild']
        unsupported = []
        for field in unsupported_fields:
            for config in configs.values():
                if field in config:
                    unsupported += ['{} not supported (target {}).'.format(field, spec['target_name'])]
        if unsupported:
            raise Exception('\n'.join(unsupported))

    def GetExtension(self):
        """Returns the extension for the target, with no leading dot.

        Uses 'product_extension' if specified, otherwise uses MSVS defaults based on
        the target type.
        """
        ext = self.spec.get('product_extension', None)
        if ext:
            return ext
        return gyp.MSVSUtil.TARGET_TYPE_EXT.get(self.spec['type'], '')

    def GetVSMacroEnv(self, base_to_build=None, config=None):
        """Get a dict of variables mapping internal VS macro names to their gyp
        equivalents."""
        target_arch = self.GetArch(config)
        if target_arch == 'x86':
            target_platform = 'Win32'
        else:
            target_platform = target_arch
        target_name = self.spec.get('product_prefix', '') + self.spec.get('product_name', self.spec['target_name'])
        target_dir = base_to_build + '\\' if base_to_build else ''
        target_ext = '.' + self.GetExtension()
        target_file_name = target_name + target_ext
        replacements = {'$(InputName)': '${root}', '$(InputPath)': '${source}', '$(IntDir)': '$!INTERMEDIATE_DIR', '$(OutDir)\\': target_dir, '$(PlatformName)': target_platform, '$(ProjectDir)\\': '', '$(ProjectName)': self.spec['target_name'], '$(TargetDir)\\': target_dir, '$(TargetExt)': target_ext, '$(TargetFileName)': target_file_name, '$(TargetName)': target_name, '$(TargetPath)': os.path.join(target_dir, target_file_name)}
        replacements.update(GetGlobalVSMacroEnv(self.vs_version))
        return replacements

    def ConvertVSMacros(self, s, base_to_build=None, config=None):
        """Convert from VS macro names to something equivalent."""
        env = self.GetVSMacroEnv(base_to_build, config=config)
        return ExpandMacros(s, env)

    def AdjustLibraries(self, libraries):
        """Strip -l from library if it's specified with that."""
        libs = [lib[2:] if lib.startswith('-l') else lib for lib in libraries]
        return [lib + '.lib' if not lib.lower().endswith('.lib') and (not lib.lower().endswith('.obj')) else lib for lib in libs]

    def _GetAndMunge(self, field, path, default, prefix, append, map):
        """Retrieve a value from |field| at |path| or return |default|. If
        |append| is specified, and the item is found, it will be appended to that
        object instead of returned. If |map| is specified, results will be
        remapped through |map| before being returned or appended."""
        result = _GenericRetrieve(field, default, path)
        result = _DoRemapping(result, map)
        result = _AddPrefix(result, prefix)
        return _AppendOrReturn(append, result)

    class _GetWrapper:

        def __init__(self, parent, field, base_path, append=None):
            self.parent = parent
            self.field = field
            self.base_path = [base_path]
            self.append = append

        def __call__(self, name, map=None, prefix='', default=None):
            return self.parent._GetAndMunge(self.field, self.base_path + [name], default=default, prefix=prefix, append=self.append, map=map)

    def GetArch(self, config):
        """Get architecture based on msvs_configuration_platform and
        msvs_target_platform. Returns either 'x86' or 'x64'."""
        configuration_platform = self.msvs_configuration_platform.get(config, '')
        platform = self.msvs_target_platform.get(config, '')
        if not platform:
            platform = configuration_platform
        return {'Win32': 'x86', 'x64': 'x64', 'ARM64': 'arm64'}.get(platform, 'x86')

    def _TargetConfig(self, config):
        """Returns the target-specific configuration."""
        if int(self.vs_version.short_name) >= 2015:
            return config
        arch = self.GetArch(config)
        if arch == 'x64' and (not config.endswith('_x64')):
            config += '_x64'
        if arch == 'x86' and config.endswith('_x64'):
            config = config.rsplit('_', 1)[0]
        return config

    def _Setting(self, path, config, default=None, prefix='', append=None, map=None):
        """_GetAndMunge for msvs_settings."""
        return self._GetAndMunge(self.msvs_settings[config], path, default, prefix, append, map)

    def _ConfigAttrib(self, path, config, default=None, prefix='', append=None, map=None):
        """_GetAndMunge for msvs_configuration_attributes."""
        return self._GetAndMunge(self.msvs_configuration_attributes[config], path, default, prefix, append, map)

    def AdjustIncludeDirs(self, include_dirs, config):
        """Updates include_dirs to expand VS specific paths, and adds the system
        include dirs used for platform SDK and similar."""
        config = self._TargetConfig(config)
        includes = include_dirs + self.msvs_system_include_dirs[config]
        includes.extend(self._Setting(('VCCLCompilerTool', 'AdditionalIncludeDirectories'), config, default=[]))
        return [self.ConvertVSMacros(p, config=config) for p in includes]

    def AdjustMidlIncludeDirs(self, midl_include_dirs, config):
        """Updates midl_include_dirs to expand VS specific paths, and adds the
        system include dirs used for platform SDK and similar."""
        config = self._TargetConfig(config)
        includes = midl_include_dirs + self.msvs_system_include_dirs[config]
        includes.extend(self._Setting(('VCMIDLTool', 'AdditionalIncludeDirectories'), config, default=[]))
        return [self.ConvertVSMacros(p, config=config) for p in includes]

    def GetComputedDefines(self, config):
        """Returns the set of defines that are injected to the defines list based
        on other VS settings."""
        config = self._TargetConfig(config)
        defines = []
        if self._ConfigAttrib(['CharacterSet'], config) == '1':
            defines.extend(('_UNICODE', 'UNICODE'))
        if self._ConfigAttrib(['CharacterSet'], config) == '2':
            defines.append('_MBCS')
        defines.extend(self._Setting(('VCCLCompilerTool', 'PreprocessorDefinitions'), config, default=[]))
        return defines

    def GetCompilerPdbName(self, config, expand_special):
        """Get the pdb file name that should be used for compiler invocations, or
        None if there's no explicit name specified."""
        config = self._TargetConfig(config)
        pdbname = self._Setting(('VCCLCompilerTool', 'ProgramDataBaseFileName'), config)
        if pdbname:
            pdbname = expand_special(self.ConvertVSMacros(pdbname))
        return pdbname

    def GetMapFileName(self, config, expand_special):
        """Gets the explicitly overridden map file name for a target or returns None
        if it's not set."""
        config = self._TargetConfig(config)
        map_file = self._Setting(('VCLinkerTool', 'MapFileName'), config)
        if map_file:
            map_file = expand_special(self.ConvertVSMacros(map_file, config=config))
        return map_file

    def GetOutputName(self, config, expand_special):
        """Gets the explicitly overridden output name for a target or returns None
        if it's not overridden."""
        config = self._TargetConfig(config)
        type = self.spec['type']
        root = 'VCLibrarianTool' if type == 'static_library' else 'VCLinkerTool'
        output_file = self._Setting((root, 'OutputFile'), config)
        if output_file:
            output_file = expand_special(self.ConvertVSMacros(output_file, config=config))
        return output_file

    def GetPDBName(self, config, expand_special, default):
        """Gets the explicitly overridden pdb name for a target or returns
        default if it's not overridden, or if no pdb will be generated."""
        config = self._TargetConfig(config)
        output_file = self._Setting(('VCLinkerTool', 'ProgramDatabaseFile'), config)
        generate_debug_info = self._Setting(('VCLinkerTool', 'GenerateDebugInformation'), config)
        if generate_debug_info == 'true':
            if output_file:
                return expand_special(self.ConvertVSMacros(output_file, config=config))
            else:
                return default
        else:
            return None

    def GetNoImportLibrary(self, config):
        """If NoImportLibrary: true, ninja will not expect the output to include
        an import library."""
        config = self._TargetConfig(config)
        noimplib = self._Setting(('NoImportLibrary',), config)
        return noimplib == 'true'

    def GetAsmflags(self, config):
        """Returns the flags that need to be added to ml invocations."""
        config = self._TargetConfig(config)
        asmflags = []
        safeseh = self._Setting(('MASM', 'UseSafeExceptionHandlers'), config)
        if safeseh == 'true':
            asmflags.append('/safeseh')
        return asmflags

    def GetCflags(self, config):
        """Returns the flags that need to be added to .c and .cc compilations."""
        config = self._TargetConfig(config)
        cflags = []
        cflags.extend(['/wd' + w for w in self.msvs_disabled_warnings[config]])
        cl = self._GetWrapper(self, self.msvs_settings[config], 'VCCLCompilerTool', append=cflags)
        cl('Optimization', map={'0': 'd', '1': '1', '2': '2', '3': 'x'}, prefix='/O', default='2')
        cl('InlineFunctionExpansion', prefix='/Ob')
        cl('DisableSpecificWarnings', prefix='/wd')
        cl('StringPooling', map={'true': '/GF'})
        cl('EnableFiberSafeOptimizations', map={'true': '/GT'})
        cl('OmitFramePointers', map={'false': '-', 'true': ''}, prefix='/Oy')
        cl('EnableIntrinsicFunctions', map={'false': '-', 'true': ''}, prefix='/Oi')
        cl('FavorSizeOrSpeed', map={'1': 't', '2': 's'}, prefix='/O')
        cl('FloatingPointModel', map={'0': 'precise', '1': 'strict', '2': 'fast'}, prefix='/fp:', default='0')
        cl('CompileAsManaged', map={'false': '', 'true': '/clr'})
        cl('WholeProgramOptimization', map={'true': '/GL'})
        cl('WarningLevel', prefix='/W')
        cl('WarnAsError', map={'true': '/WX'})
        cl('CallingConvention', map={'0': 'd', '1': 'r', '2': 'z', '3': 'v'}, prefix='/G')
        cl('DebugInformationFormat', map={'1': '7', '3': 'i', '4': 'I'}, prefix='/Z')
        cl('RuntimeTypeInfo', map={'true': '/GR', 'false': '/GR-'})
        cl('EnableFunctionLevelLinking', map={'true': '/Gy', 'false': '/Gy-'})
        cl('MinimalRebuild', map={'true': '/Gm'})
        cl('BufferSecurityCheck', map={'true': '/GS', 'false': '/GS-'})
        cl('BasicRuntimeChecks', map={'1': 's', '2': 'u', '3': '1'}, prefix='/RTC')
        cl('RuntimeLibrary', map={'0': 'T', '1': 'Td', '2': 'D', '3': 'Dd'}, prefix='/M')
        cl('ExceptionHandling', map={'1': 'sc', '2': 'a'}, prefix='/EH')
        cl('DefaultCharIsUnsigned', map={'true': '/J'})
        cl('TreatWChar_tAsBuiltInType', map={'false': '-', 'true': ''}, prefix='/Zc:wchar_t')
        cl('EnablePREfast', map={'true': '/analyze'})
        cl('AdditionalOptions', prefix='')
        cl('EnableEnhancedInstructionSet', map={'1': 'SSE', '2': 'SSE2', '3': 'AVX', '4': 'IA32', '5': 'AVX2'}, prefix='/arch:')
        cflags.extend(['/FI' + f for f in self._Setting(('VCCLCompilerTool', 'ForcedIncludeFiles'), config, default=[])])
        if float(self.vs_version.project_version) >= 12.0:
            cflags.append('/FS')
        cflags = [x for x in cflags if not x.startswith('/MP')]
        return cflags

    def _GetPchFlags(self, config, extension):
        """Get the flags to be added to the cflags for precompiled header support."""
        config = self._TargetConfig(config)
        if self.msvs_precompiled_header[config]:
            source_ext = os.path.splitext(self.msvs_precompiled_source[config])[1]
            if _LanguageMatchesForPch(source_ext, extension):
                pch = self.msvs_precompiled_header[config]
                pchbase = os.path.split(pch)[1]
                return ['/Yu' + pch, '/FI' + pch, '/Fp${pchprefix}.' + pchbase + '.pch']
        return []

    def GetCflagsC(self, config):
        """Returns the flags that need to be added to .c compilations."""
        config = self._TargetConfig(config)
        return self._GetPchFlags(config, '.c')

    def GetCflagsCC(self, config):
        """Returns the flags that need to be added to .cc compilations."""
        config = self._TargetConfig(config)
        return ['/TP'] + self._GetPchFlags(config, '.cc')

    def _GetAdditionalLibraryDirectories(self, root, config, gyp_to_build_path):
        """Get and normalize the list of paths in AdditionalLibraryDirectories
        setting."""
        config = self._TargetConfig(config)
        libpaths = self._Setting((root, 'AdditionalLibraryDirectories'), config, default=[])
        libpaths = [os.path.normpath(gyp_to_build_path(self.ConvertVSMacros(p, config=config))) for p in libpaths]
        return ['/LIBPATH:"' + p + '"' for p in libpaths]

    def GetLibFlags(self, config, gyp_to_build_path):
        """Returns the flags that need to be added to lib commands."""
        config = self._TargetConfig(config)
        libflags = []
        lib = self._GetWrapper(self, self.msvs_settings[config], 'VCLibrarianTool', append=libflags)
        libflags.extend(self._GetAdditionalLibraryDirectories('VCLibrarianTool', config, gyp_to_build_path))
        lib('LinkTimeCodeGeneration', map={'true': '/LTCG'})
        lib('TargetMachine', map={'1': 'X86', '17': 'X64', '3': 'ARM'}, prefix='/MACHINE:')
        lib('AdditionalOptions')
        return libflags

    def GetDefFile(self, gyp_to_build_path):
        """Returns the .def file from sources, if any.  Otherwise returns None."""
        spec = self.spec
        if spec['type'] in ('shared_library', 'loadable_module', 'executable'):
            def_files = [s for s in spec.get('sources', []) if s.lower().endswith('.def')]
            if len(def_files) == 1:
                return gyp_to_build_path(def_files[0])
            elif len(def_files) > 1:
                raise Exception('Multiple .def files')
        return None

    def _GetDefFileAsLdflags(self, ldflags, gyp_to_build_path):
        """.def files get implicitly converted to a ModuleDefinitionFile for the
        linker in the VS generator. Emulate that behaviour here."""
        def_file = self.GetDefFile(gyp_to_build_path)
        if def_file:
            ldflags.append('/DEF:"%s"' % def_file)

    def GetPGDName(self, config, expand_special):
        """Gets the explicitly overridden pgd name for a target or returns None
        if it's not overridden."""
        config = self._TargetConfig(config)
        output_file = self._Setting(('VCLinkerTool', 'ProfileGuidedDatabase'), config)
        if output_file:
            output_file = expand_special(self.ConvertVSMacros(output_file, config=config))
        return output_file

    def GetLdflags(self, config, gyp_to_build_path, expand_special, manifest_base_name, output_name, is_executable, build_dir):
        """Returns the flags that need to be added to link commands, and the
        manifest files."""
        config = self._TargetConfig(config)
        ldflags = []
        ld = self._GetWrapper(self, self.msvs_settings[config], 'VCLinkerTool', append=ldflags)
        self._GetDefFileAsLdflags(ldflags, gyp_to_build_path)
        ld('GenerateDebugInformation', map={'true': '/DEBUG'})
        ld('TargetMachine', map={'1': 'X86', '17': 'X64', '3': 'ARM', '18': 'ARM64'}, prefix='/MACHINE:')
        ldflags.extend(self._GetAdditionalLibraryDirectories('VCLinkerTool', config, gyp_to_build_path))
        ld('DelayLoadDLLs', prefix='/DELAYLOAD:')
        ld('TreatLinkerWarningAsErrors', prefix='/WX', map={'true': '', 'false': ':NO'})
        out = self.GetOutputName(config, expand_special)
        if out:
            ldflags.append('/OUT:' + out)
        pdb = self.GetPDBName(config, expand_special, output_name + '.pdb')
        if pdb:
            ldflags.append('/PDB:' + pdb)
        pgd = self.GetPGDName(config, expand_special)
        if pgd:
            ldflags.append('/PGD:' + pgd)
        map_file = self.GetMapFileName(config, expand_special)
        ld('GenerateMapFile', map={'true': '/MAP:' + map_file if map_file else '/MAP'})
        ld('MapExports', map={'true': '/MAPINFO:EXPORTS'})
        ld('AdditionalOptions', prefix='')
        minimum_required_version = self._Setting(('VCLinkerTool', 'MinimumRequiredVersion'), config, default='')
        if minimum_required_version:
            minimum_required_version = ',' + minimum_required_version
        ld('SubSystem', map={'1': 'CONSOLE%s' % minimum_required_version, '2': 'WINDOWS%s' % minimum_required_version}, prefix='/SUBSYSTEM:')
        stack_reserve_size = self._Setting(('VCLinkerTool', 'StackReserveSize'), config, default='')
        if stack_reserve_size:
            stack_commit_size = self._Setting(('VCLinkerTool', 'StackCommitSize'), config, default='')
            if stack_commit_size:
                stack_commit_size = ',' + stack_commit_size
            ldflags.append(f'/STACK:{stack_reserve_size}{stack_commit_size}')
        ld('TerminalServerAware', map={'1': ':NO', '2': ''}, prefix='/TSAWARE')
        ld('LinkIncremental', map={'1': ':NO', '2': ''}, prefix='/INCREMENTAL')
        ld('BaseAddress', prefix='/BASE:')
        ld('FixedBaseAddress', map={'1': ':NO', '2': ''}, prefix='/FIXED')
        ld('RandomizedBaseAddress', map={'1': ':NO', '2': ''}, prefix='/DYNAMICBASE')
        ld('DataExecutionPrevention', map={'1': ':NO', '2': ''}, prefix='/NXCOMPAT')
        ld('OptimizeReferences', map={'1': 'NOREF', '2': 'REF'}, prefix='/OPT:')
        ld('ForceSymbolReferences', prefix='/INCLUDE:')
        ld('EnableCOMDATFolding', map={'1': 'NOICF', '2': 'ICF'}, prefix='/OPT:')
        ld('LinkTimeCodeGeneration', map={'1': '', '2': ':PGINSTRUMENT', '3': ':PGOPTIMIZE', '4': ':PGUPDATE'}, prefix='/LTCG')
        ld('IgnoreDefaultLibraryNames', prefix='/NODEFAULTLIB:')
        ld('ResourceOnlyDLL', map={'true': '/NOENTRY'})
        ld('EntryPointSymbol', prefix='/ENTRY:')
        ld('Profile', map={'true': '/PROFILE'})
        ld('LargeAddressAware', map={'1': ':NO', '2': ''}, prefix='/LARGEADDRESSAWARE')
        ld('AdditionalDependencies', prefix='')
        if self.GetArch(config) == 'x86':
            safeseh_default = 'true'
        else:
            safeseh_default = None
        ld('ImageHasSafeExceptionHandlers', map={'false': ':NO', 'true': ''}, prefix='/SAFESEH', default=safeseh_default)
        if not any(('DYNAMICBASE' in flag or flag == '/FIXED' for flag in ldflags)):
            ldflags.append('/DYNAMICBASE')
        if not any(('NXCOMPAT' in flag for flag in ldflags)):
            ldflags.append('/NXCOMPAT')
        have_def_file = any((flag.startswith('/DEF:') for flag in ldflags))
        manifest_flags, intermediate_manifest, manifest_files = self._GetLdManifestFlags(config, manifest_base_name, gyp_to_build_path, is_executable and (not have_def_file), build_dir)
        ldflags.extend(manifest_flags)
        return (ldflags, intermediate_manifest, manifest_files)

    def _GetLdManifestFlags(self, config, name, gyp_to_build_path, allow_isolation, build_dir):
        """Returns a 3-tuple:
        - the set of flags that need to be added to the link to generate
          a default manifest
        - the intermediate manifest that the linker will generate that should be
          used to assert it doesn't add anything to the merged one.
        - the list of all the manifest files to be merged by the manifest tool and
          included into the link."""
        generate_manifest = self._Setting(('VCLinkerTool', 'GenerateManifest'), config, default='true')
        if generate_manifest != 'true':
            return (['/MANIFEST:NO'], [], [])
        output_name = name + '.intermediate.manifest'
        flags = ['/MANIFEST', '/ManifestFile:' + output_name]
        flags.append('/MANIFESTUAC:NO')
        config = self._TargetConfig(config)
        enable_uac = self._Setting(('VCLinkerTool', 'EnableUAC'), config, default='true')
        manifest_files = []
        generated_manifest_outer = "<?xml version='1.0' encoding='UTF-8' standalone='yes'?><assembly xmlns='urn:schemas-microsoft-com:asm.v1' manifestVersion='1.0'>%s</assembly>"
        if enable_uac == 'true':
            execution_level = self._Setting(('VCLinkerTool', 'UACExecutionLevel'), config, default='0')
            execution_level_map = {'0': 'asInvoker', '1': 'highestAvailable', '2': 'requireAdministrator'}
            ui_access = self._Setting(('VCLinkerTool', 'UACUIAccess'), config, default='false')
            inner = '\n<trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n  <security>\n    <requestedPrivileges>\n      <requestedExecutionLevel level=\'{}\' uiAccess=\'{}\' />\n    </requestedPrivileges>\n  </security>\n</trustInfo>'.format(execution_level_map[execution_level], ui_access)
        else:
            inner = ''
        generated_manifest_contents = generated_manifest_outer % inner
        generated_name = name + '.generated.manifest'
        build_dir_generated_name = os.path.join(build_dir, generated_name)
        gyp.common.EnsureDirExists(build_dir_generated_name)
        f = gyp.common.WriteOnDiff(build_dir_generated_name)
        f.write(generated_manifest_contents)
        f.close()
        manifest_files = [generated_name]
        if allow_isolation:
            flags.append('/ALLOWISOLATION')
        manifest_files += self._GetAdditionalManifestFiles(config, gyp_to_build_path)
        return (flags, output_name, manifest_files)

    def _GetAdditionalManifestFiles(self, config, gyp_to_build_path):
        """Gets additional manifest files that are added to the default one
        generated by the linker."""
        files = self._Setting(('VCManifestTool', 'AdditionalManifestFiles'), config, default=[])
        if isinstance(files, str):
            files = files.split(';')
        return [os.path.normpath(gyp_to_build_path(self.ConvertVSMacros(f, config=config))) for f in files]

    def IsUseLibraryDependencyInputs(self, config):
        """Returns whether the target should be linked via Use Library Dependency
        Inputs (using component .objs of a given .lib)."""
        config = self._TargetConfig(config)
        uldi = self._Setting(('VCLinkerTool', 'UseLibraryDependencyInputs'), config)
        return uldi == 'true'

    def IsEmbedManifest(self, config):
        """Returns whether manifest should be linked into binary."""
        config = self._TargetConfig(config)
        embed = self._Setting(('VCManifestTool', 'EmbedManifest'), config, default='true')
        return embed == 'true'

    def IsLinkIncremental(self, config):
        """Returns whether the target should be linked incrementally."""
        config = self._TargetConfig(config)
        link_inc = self._Setting(('VCLinkerTool', 'LinkIncremental'), config)
        return link_inc != '1'

    def GetRcflags(self, config, gyp_to_ninja_path):
        """Returns the flags that need to be added to invocations of the resource
        compiler."""
        config = self._TargetConfig(config)
        rcflags = []
        rc = self._GetWrapper(self, self.msvs_settings[config], 'VCResourceCompilerTool', append=rcflags)
        rc('AdditionalIncludeDirectories', map=gyp_to_ninja_path, prefix='/I')
        rcflags.append('/I' + gyp_to_ninja_path('.'))
        rc('PreprocessorDefinitions', prefix='/d')
        rc('Culture', prefix='/l', map=lambda x: hex(int(x))[2:])
        return rcflags

    def BuildCygwinBashCommandLine(self, args, path_to_base):
        """Build a command line that runs args via cygwin bash. We assume that all
        incoming paths are in Windows normpath'd form, so they need to be
        converted to posix style for the part of the command line that's passed to
        bash. We also have to do some Visual Studio macro emulation here because
        various rules use magic VS names for things. Also note that rules that
        contain ninja variables cannot be fixed here (for example ${source}), so
        the outer generator needs to make sure that the paths that are written out
        are in posix style, if the command line will be used here."""
        cygwin_dir = os.path.normpath(os.path.join(path_to_base, self.msvs_cygwin_dirs[0]))
        cd = ('cd %s' % path_to_base).replace('\\', '/')
        args = [a.replace('\\', '/').replace('"', '\\"') for a in args]
        args = ["'%s'" % a.replace("'", "'\\''") for a in args]
        bash_cmd = ' '.join(args)
        cmd = 'call "%s\\setup_env.bat" && set CYGWIN=nontsec && ' % cygwin_dir + f'bash -c "{cd} ; {bash_cmd}"'
        return cmd
    RuleShellFlags = collections.namedtuple('RuleShellFlags', ['cygwin', 'quote'])

    def GetRuleShellFlags(self, rule):
        """Return RuleShellFlags about how the given rule should be run. This
        includes whether it should run under cygwin (msvs_cygwin_shell), and
        whether the commands should be quoted (msvs_quote_cmd)."""
        cygwin = int(rule.get('msvs_cygwin_shell', self.spec.get('msvs_cygwin_shell', 1))) != 0
        quote_cmd = int(rule.get('msvs_quote_cmd', 1))
        assert quote_cmd != 0 or cygwin != 1, 'msvs_quote_cmd=0 only applicable for msvs_cygwin_shell=0'
        return MsvsSettings.RuleShellFlags(cygwin, quote_cmd)

    def _HasExplicitRuleForExtension(self, spec, extension):
        """Determine if there's an explicit rule for a particular extension."""
        for rule in spec.get('rules', []):
            if rule['extension'] == extension:
                return True
        return False

    def _HasExplicitIdlActions(self, spec):
        """Determine if an action should not run midl for .idl files."""
        return any([action.get('explicit_idl_action', 0) for action in spec.get('actions', [])])

    def HasExplicitIdlRulesOrActions(self, spec):
        """Determine if there's an explicit rule or action for idl files. When
        there isn't we need to generate implicit rules to build MIDL .idl files."""
        return self._HasExplicitRuleForExtension(spec, 'idl') or self._HasExplicitIdlActions(spec)

    def HasExplicitAsmRules(self, spec):
        """Determine if there's an explicit rule for asm files. When there isn't we
        need to generate implicit rules to assemble .asm files."""
        return self._HasExplicitRuleForExtension(spec, 'asm')

    def GetIdlBuildData(self, source, config):
        """Determine the implicit outputs for an idl file. Returns output
        directory, outputs, and variables and flags that are required."""
        config = self._TargetConfig(config)
        midl_get = self._GetWrapper(self, self.msvs_settings[config], 'VCMIDLTool')

        def midl(name, default=None):
            return self.ConvertVSMacros(midl_get(name, default=default), config=config)
        tlb = midl('TypeLibraryName', default='${root}.tlb')
        header = midl('HeaderFileName', default='${root}.h')
        dlldata = midl('DLLDataFileName', default='dlldata.c')
        iid = midl('InterfaceIdentifierFileName', default='${root}_i.c')
        proxy = midl('ProxyFileName', default='${root}_p.c')
        outdir = midl('OutputDirectory', default='')
        output = [header, dlldata, iid, proxy]
        variables = [('tlb', tlb), ('h', header), ('dlldata', dlldata), ('iid', iid), ('proxy', proxy)]
        target_platform = self.GetArch(config)
        if target_platform == 'x86':
            target_platform = 'win32'
        flags = ['/char', 'signed', '/env', target_platform, '/Oicf']
        return (outdir, output, variables, flags)