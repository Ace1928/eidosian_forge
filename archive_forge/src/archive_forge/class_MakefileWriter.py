import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
class MakefileWriter:
    """MakefileWriter packages up the writing of one target-specific foobar.mk.

    Its only real entry point is Write(), and is mostly used for namespacing.
    """

    def __init__(self, generator_flags, flavor):
        self.generator_flags = generator_flags
        self.flavor = flavor
        self.suffix_rules_srcdir = {}
        self.suffix_rules_objdir1 = {}
        self.suffix_rules_objdir2 = {}
        for ext in COMPILABLE_EXTENSIONS.keys():
            self.suffix_rules_srcdir.update({ext: '$(obj).$(TOOLSET)/$(TARGET)/%%.o: $(srcdir)/%%%s FORCE_DO_CMD\n\t@$(call do_cmd,%s,1)\n' % (ext, COMPILABLE_EXTENSIONS[ext])})
            self.suffix_rules_objdir1.update({ext: '$(obj).$(TOOLSET)/$(TARGET)/%%.o: $(obj).$(TOOLSET)/%%%s FORCE_DO_CMD\n\t@$(call do_cmd,%s,1)\n' % (ext, COMPILABLE_EXTENSIONS[ext])})
            self.suffix_rules_objdir2.update({ext: '$(obj).$(TOOLSET)/$(TARGET)/%%.o: $(obj)/%%%s FORCE_DO_CMD\n\t@$(call do_cmd,%s,1)\n' % (ext, COMPILABLE_EXTENSIONS[ext])})

    def Write(self, qualified_target, base_path, output_filename, spec, configs, part_of_all):
        """The main entry point: writes a .mk file for a single target.

        Arguments:
          qualified_target: target we're generating
          base_path: path relative to source root we're building in, used to resolve
                     target-relative paths
          output_filename: output .mk file name to write
          spec, configs: gyp info
          part_of_all: flag indicating this target is part of 'all'
        """
        gyp.common.EnsureDirExists(output_filename)
        self.fp = open(output_filename, 'w')
        self.fp.write(header)
        self.qualified_target = qualified_target
        self.path = base_path
        self.target = spec['target_name']
        self.type = spec['type']
        self.toolset = spec['toolset']
        self.is_mac_bundle = gyp.xcode_emulation.IsMacBundle(self.flavor, spec)
        if self.flavor == 'mac':
            self.xcode_settings = gyp.xcode_emulation.XcodeSettings(spec)
        else:
            self.xcode_settings = None
        deps, link_deps = self.ComputeDeps(spec)
        extra_outputs = []
        extra_sources = []
        extra_link_deps = []
        extra_mac_bundle_resources = []
        mac_bundle_deps = []
        if self.is_mac_bundle:
            self.output = self.ComputeMacBundleOutput(spec)
            self.output_binary = self.ComputeMacBundleBinaryOutput(spec)
        else:
            self.output = self.output_binary = self.ComputeOutput(spec)
        self.is_standalone_static_library = bool(spec.get('standalone_static_library', 0))
        self._INSTALLABLE_TARGETS = ('executable', 'loadable_module', 'shared_library')
        if self.is_standalone_static_library or self.type in self._INSTALLABLE_TARGETS:
            self.alias = os.path.basename(self.output)
            install_path = self._InstallableTargetInstallPath()
        else:
            self.alias = self.output
            install_path = self.output
        self.WriteLn('TOOLSET := ' + self.toolset)
        self.WriteLn('TARGET := ' + self.target)
        if 'actions' in spec:
            self.WriteActions(spec['actions'], extra_sources, extra_outputs, extra_mac_bundle_resources, part_of_all)
        if 'rules' in spec:
            self.WriteRules(spec['rules'], extra_sources, extra_outputs, extra_mac_bundle_resources, part_of_all)
        if 'copies' in spec:
            self.WriteCopies(spec['copies'], extra_outputs, part_of_all)
        if self.is_mac_bundle:
            all_mac_bundle_resources = spec.get('mac_bundle_resources', []) + extra_mac_bundle_resources
            self.WriteMacBundleResources(all_mac_bundle_resources, mac_bundle_deps)
            self.WriteMacInfoPlist(mac_bundle_deps)
        all_sources = spec.get('sources', []) + extra_sources
        if all_sources:
            self.WriteSources(configs, deps, all_sources, extra_outputs, extra_link_deps, part_of_all, gyp.xcode_emulation.MacPrefixHeader(self.xcode_settings, lambda p: Sourceify(self.Absolutify(p)), self.Pchify))
            sources = [x for x in all_sources if Compilable(x)]
            if sources:
                self.WriteLn(SHARED_HEADER_SUFFIX_RULES_COMMENT1)
                extensions = {os.path.splitext(s)[1] for s in sources}
                for ext in extensions:
                    if ext in self.suffix_rules_srcdir:
                        self.WriteLn(self.suffix_rules_srcdir[ext])
                self.WriteLn(SHARED_HEADER_SUFFIX_RULES_COMMENT2)
                for ext in extensions:
                    if ext in self.suffix_rules_objdir1:
                        self.WriteLn(self.suffix_rules_objdir1[ext])
                for ext in extensions:
                    if ext in self.suffix_rules_objdir2:
                        self.WriteLn(self.suffix_rules_objdir2[ext])
                self.WriteLn('# End of this set of suffix rules')
                if self.is_mac_bundle:
                    mac_bundle_deps.append(self.output_binary)
        self.WriteTarget(spec, configs, deps, extra_link_deps + link_deps, mac_bundle_deps, extra_outputs, part_of_all)
        target_outputs[qualified_target] = install_path
        if self.type in ('static_library', 'shared_library'):
            target_link_deps[qualified_target] = self.output_binary
        if self.generator_flags.get('android_ndk_version', None):
            self.WriteAndroidNdkModuleRule(self.target, all_sources, link_deps)
        self.fp.close()

    def WriteSubMake(self, output_filename, makefile_path, targets, build_dir):
        """Write a "sub-project" Makefile.

        This is a small, wrapper Makefile that calls the top-level Makefile to build
        the targets from a single gyp file (i.e. a sub-project).

        Arguments:
          output_filename: sub-project Makefile name to write
          makefile_path: path to the top-level Makefile
          targets: list of "all" targets for this sub-project
          build_dir: build output directory, relative to the sub-project
        """
        gyp.common.EnsureDirExists(output_filename)
        self.fp = open(output_filename, 'w')
        self.fp.write(header)
        self.WriteLn('export builddir_name ?= %s' % os.path.join(os.path.dirname(output_filename), build_dir))
        self.WriteLn('.PHONY: all')
        self.WriteLn('all:')
        if makefile_path:
            makefile_path = ' -C ' + makefile_path
        self.WriteLn('\t$(MAKE){} {}'.format(makefile_path, ' '.join(targets)))
        self.fp.close()

    def WriteActions(self, actions, extra_sources, extra_outputs, extra_mac_bundle_resources, part_of_all):
        """Write Makefile code for any 'actions' from the gyp input.

        extra_sources: a list that will be filled in with newly generated source
                       files, if any
        extra_outputs: a list that will be filled in with any outputs of these
                       actions (used to make other pieces dependent on these
                       actions)
        part_of_all: flag indicating this target is part of 'all'
        """
        env = self.GetSortedXcodeEnv()
        for action in actions:
            name = StringToMakefileVariable('{}_{}'.format(self.qualified_target, action['action_name']))
            self.WriteLn('### Rules for action "%s":' % action['action_name'])
            inputs = action['inputs']
            outputs = action['outputs']
            dirs = set()
            for out in outputs:
                dir = os.path.split(out)[0]
                if dir:
                    dirs.add(dir)
            if int(action.get('process_outputs_as_sources', False)):
                extra_sources += outputs
            if int(action.get('process_outputs_as_mac_bundle_resources', False)):
                extra_mac_bundle_resources += outputs
            action_commands = action['action']
            if self.flavor == 'mac':
                action_commands = [gyp.xcode_emulation.ExpandEnvVars(command, env) for command in action_commands]
            command = gyp.common.EncodePOSIXShellList(action_commands)
            if 'message' in action:
                self.WriteLn('quiet_cmd_{} = ACTION {} $@'.format(name, action['message']))
            else:
                self.WriteLn(f'quiet_cmd_{name} = ACTION {name} $@')
            if len(dirs) > 0:
                command = 'mkdir -p %s' % ' '.join(dirs) + '; ' + command
            cd_action = 'cd %s; ' % Sourceify(self.path or '.')
            command = command.replace('$(TARGET)', self.target)
            cd_action = cd_action.replace('$(TARGET)', self.target)
            self.WriteLn('cmd_%s = LD_LIBRARY_PATH=$(builddir)/lib.host:$(builddir)/lib.target:$$LD_LIBRARY_PATH; export LD_LIBRARY_PATH; %s%s' % (name, cd_action, command))
            self.WriteLn()
            outputs = [self.Absolutify(o) for o in outputs]
            self.WriteLn('%s: obj := $(abs_obj)' % QuoteSpaces(outputs[0]))
            self.WriteLn('%s: builddir := $(abs_builddir)' % QuoteSpaces(outputs[0]))
            self.WriteSortedXcodeEnv(outputs[0], self.GetSortedXcodeEnv())
            for input in inputs:
                assert ' ' not in input, 'Spaces in action input filenames not supported (%s)' % input
            for output in outputs:
                assert ' ' not in output, 'Spaces in action output filenames not supported (%s)' % output
            outputs = [gyp.xcode_emulation.ExpandEnvVars(o, env) for o in outputs]
            inputs = [gyp.xcode_emulation.ExpandEnvVars(i, env) for i in inputs]
            self.WriteDoCmd(outputs, [Sourceify(self.Absolutify(i)) for i in inputs], part_of_all=part_of_all, command=name)
            outputs_variable = 'action_%s_outputs' % name
            self.WriteLn('{} := {}'.format(outputs_variable, ' '.join(outputs)))
            extra_outputs.append('$(%s)' % outputs_variable)
            self.WriteLn()
        self.WriteLn()

    def WriteRules(self, rules, extra_sources, extra_outputs, extra_mac_bundle_resources, part_of_all):
        """Write Makefile code for any 'rules' from the gyp input.

        extra_sources: a list that will be filled in with newly generated source
                       files, if any
        extra_outputs: a list that will be filled in with any outputs of these
                       rules (used to make other pieces dependent on these rules)
        part_of_all: flag indicating this target is part of 'all'
        """
        env = self.GetSortedXcodeEnv()
        for rule in rules:
            name = StringToMakefileVariable('{}_{}'.format(self.qualified_target, rule['rule_name']))
            count = 0
            self.WriteLn('### Generated for rule %s:' % name)
            all_outputs = []
            for rule_source in rule.get('rule_sources', []):
                dirs = set()
                rule_source_dirname, rule_source_basename = os.path.split(rule_source)
                rule_source_root, rule_source_ext = os.path.splitext(rule_source_basename)
                outputs = [self.ExpandInputRoot(out, rule_source_root, rule_source_dirname) for out in rule['outputs']]
                for out in outputs:
                    dir = os.path.dirname(out)
                    if dir:
                        dirs.add(dir)
                if int(rule.get('process_outputs_as_sources', False)):
                    extra_sources += outputs
                if int(rule.get('process_outputs_as_mac_bundle_resources', False)):
                    extra_mac_bundle_resources += outputs
                inputs = [Sourceify(self.Absolutify(i)) for i in [rule_source] + rule.get('inputs', [])]
                actions = ['$(call do_cmd,%s_%d)' % (name, count)]
                if name == 'resources_grit':
                    actions += ['@touch --no-create $@']
                outputs = [gyp.xcode_emulation.ExpandEnvVars(o, env) for o in outputs]
                inputs = [gyp.xcode_emulation.ExpandEnvVars(i, env) for i in inputs]
                outputs = [self.Absolutify(o) for o in outputs]
                all_outputs += outputs
                self.WriteLn('%s: obj := $(abs_obj)' % outputs[0])
                self.WriteLn('%s: builddir := $(abs_builddir)' % outputs[0])
                self.WriteMakeRule(outputs, inputs, actions, command='%s_%d' % (name, count))
                variables_with_spaces = re.compile('\\$\\([^ ]* \\$<\\)')
                for output in outputs:
                    output = re.sub(variables_with_spaces, '', output)
                    assert ' ' not in output, 'Spaces in rule filenames not yet supported (%s)' % output
                self.WriteLn('all_deps += %s' % ' '.join(outputs))
                action = [self.ExpandInputRoot(ac, rule_source_root, rule_source_dirname) for ac in rule['action']]
                mkdirs = ''
                if len(dirs) > 0:
                    mkdirs = 'mkdir -p %s; ' % ' '.join(dirs)
                cd_action = 'cd %s; ' % Sourceify(self.path or '.')
                if self.flavor == 'mac':
                    action = [gyp.xcode_emulation.ExpandEnvVars(command, env) for command in action]
                action = gyp.common.EncodePOSIXShellList(action)
                action = action.replace('$(TARGET)', self.target)
                cd_action = cd_action.replace('$(TARGET)', self.target)
                mkdirs = mkdirs.replace('$(TARGET)', self.target)
                self.WriteLn('cmd_%(name)s_%(count)d = LD_LIBRARY_PATH=$(builddir)/lib.host:$(builddir)/lib.target:$$LD_LIBRARY_PATH; export LD_LIBRARY_PATH; %(cd_action)s%(mkdirs)s%(action)s' % {'action': action, 'cd_action': cd_action, 'count': count, 'mkdirs': mkdirs, 'name': name})
                self.WriteLn('quiet_cmd_%(name)s_%(count)d = RULE %(name)s_%(count)d $@' % {'count': count, 'name': name})
                self.WriteLn()
                count += 1
            outputs_variable = 'rule_%s_outputs' % name
            self.WriteList(all_outputs, outputs_variable)
            extra_outputs.append('$(%s)' % outputs_variable)
            self.WriteLn('### Finished generating for rule: %s' % name)
            self.WriteLn()
        self.WriteLn('### Finished generating for all rules')
        self.WriteLn('')

    def WriteCopies(self, copies, extra_outputs, part_of_all):
        """Write Makefile code for any 'copies' from the gyp input.

        extra_outputs: a list that will be filled in with any outputs of this action
                       (used to make other pieces dependent on this action)
        part_of_all: flag indicating this target is part of 'all'
        """
        self.WriteLn('### Generated for copy rule.')
        variable = StringToMakefileVariable(self.qualified_target + '_copies')
        outputs = []
        for copy in copies:
            for path in copy['files']:
                path = Sourceify(self.Absolutify(path))
                filename = os.path.split(path)[1]
                output = Sourceify(self.Absolutify(os.path.join(copy['destination'], filename)))
                env = self.GetSortedXcodeEnv()
                output = gyp.xcode_emulation.ExpandEnvVars(output, env)
                path = gyp.xcode_emulation.ExpandEnvVars(path, env)
                self.WriteDoCmd([output], [path], 'copy', part_of_all)
                outputs.append(output)
        self.WriteLn('{} = {}'.format(variable, ' '.join((QuoteSpaces(o) for o in outputs))))
        extra_outputs.append('$(%s)' % variable)
        self.WriteLn()

    def WriteMacBundleResources(self, resources, bundle_deps):
        """Writes Makefile code for 'mac_bundle_resources'."""
        self.WriteLn('### Generated for mac_bundle_resources')
        for output, res in gyp.xcode_emulation.GetMacBundleResources(generator_default_variables['PRODUCT_DIR'], self.xcode_settings, [Sourceify(self.Absolutify(r)) for r in resources]):
            _, ext = os.path.splitext(output)
            if ext != '.xcassets':
                self.WriteDoCmd([output], [res], 'mac_tool,,,copy-bundle-resource', part_of_all=True)
                bundle_deps.append(output)

    def WriteMacInfoPlist(self, bundle_deps):
        """Write Makefile code for bundle Info.plist files."""
        info_plist, out, defines, extra_env = gyp.xcode_emulation.GetMacInfoPlist(generator_default_variables['PRODUCT_DIR'], self.xcode_settings, lambda p: Sourceify(self.Absolutify(p)))
        if not info_plist:
            return
        if defines:
            intermediate_plist = '$(obj).$(TOOLSET)/$(TARGET)/' + os.path.basename(info_plist)
            self.WriteList(defines, intermediate_plist + ': INFOPLIST_DEFINES', '-D', quoter=EscapeCppDefine)
            self.WriteMakeRule([intermediate_plist], [info_plist], ['$(call do_cmd,infoplist)', '@plutil -convert xml1 $@ $@'])
            info_plist = intermediate_plist
        self.WriteSortedXcodeEnv(out, self.GetSortedXcodeEnv(additional_settings=extra_env))
        self.WriteDoCmd([out], [info_plist], 'mac_tool,,,copy-info-plist', part_of_all=True)
        bundle_deps.append(out)

    def WriteSources(self, configs, deps, sources, extra_outputs, extra_link_deps, part_of_all, precompiled_header):
        """Write Makefile code for any 'sources' from the gyp input.
        These are source files necessary to build the current target.

        configs, deps, sources: input from gyp.
        extra_outputs: a list of extra outputs this action should be dependent on;
                       used to serialize action/rules before compilation
        extra_link_deps: a list that will be filled in with any outputs of
                         compilation (to be used in link lines)
        part_of_all: flag indicating this target is part of 'all'
        """
        for configname in sorted(configs.keys()):
            config = configs[configname]
            self.WriteList(config.get('defines'), 'DEFS_%s' % configname, prefix='-D', quoter=EscapeCppDefine)
            if self.flavor == 'mac':
                cflags = self.xcode_settings.GetCflags(configname, arch=config.get('xcode_configuration_platform'))
                cflags_c = self.xcode_settings.GetCflagsC(configname)
                cflags_cc = self.xcode_settings.GetCflagsCC(configname)
                cflags_objc = self.xcode_settings.GetCflagsObjC(configname)
                cflags_objcc = self.xcode_settings.GetCflagsObjCC(configname)
            else:
                cflags = config.get('cflags')
                cflags_c = config.get('cflags_c')
                cflags_cc = config.get('cflags_cc')
            self.WriteLn('# Flags passed to all source files.')
            self.WriteList(cflags, 'CFLAGS_%s' % configname)
            self.WriteLn('# Flags passed to only C files.')
            self.WriteList(cflags_c, 'CFLAGS_C_%s' % configname)
            self.WriteLn('# Flags passed to only C++ files.')
            self.WriteList(cflags_cc, 'CFLAGS_CC_%s' % configname)
            if self.flavor == 'mac':
                self.WriteLn('# Flags passed to only ObjC files.')
                self.WriteList(cflags_objc, 'CFLAGS_OBJC_%s' % configname)
                self.WriteLn('# Flags passed to only ObjC++ files.')
                self.WriteList(cflags_objcc, 'CFLAGS_OBJCC_%s' % configname)
            includes = config.get('include_dirs')
            if includes:
                includes = [Sourceify(self.Absolutify(i)) for i in includes]
            self.WriteList(includes, 'INCS_%s' % configname, prefix='-I')
        compilable = list(filter(Compilable, sources))
        objs = [self.Objectify(self.Absolutify(Target(c))) for c in compilable]
        self.WriteList(objs, 'OBJS')
        for obj in objs:
            assert ' ' not in obj, 'Spaces in object filenames not supported (%s)' % obj
        self.WriteLn('# Add to the list of files we specially track dependencies for.')
        self.WriteLn('all_deps += $(OBJS)')
        self.WriteLn()
        if deps:
            self.WriteMakeRule(['$(OBJS)'], deps, comment='Make sure our dependencies are built before any of us.', order_only=True)
        if extra_outputs:
            self.WriteMakeRule(['$(OBJS)'], extra_outputs, comment='Make sure our actions/rules run before any of us.', order_only=True)
        pchdeps = precompiled_header.GetObjDependencies(compilable, objs)
        if pchdeps:
            self.WriteLn('# Dependencies from obj files to their precompiled headers')
            for source, obj, gch in pchdeps:
                self.WriteLn(f'{obj}: {gch}')
            self.WriteLn('# End precompiled header dependencies')
        if objs:
            extra_link_deps.append('$(OBJS)')
            self.WriteLn('# CFLAGS et al overrides must be target-local.\n# See "Target-specific Variable Values" in the GNU Make manual.')
            self.WriteLn('$(OBJS): TOOLSET := $(TOOLSET)')
            self.WriteLn('$(OBJS): GYP_CFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) %s ' % precompiled_header.GetInclude('c') + '$(CFLAGS_$(BUILDTYPE)) $(CFLAGS_C_$(BUILDTYPE))')
            self.WriteLn('$(OBJS): GYP_CXXFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) %s ' % precompiled_header.GetInclude('cc') + '$(CFLAGS_$(BUILDTYPE)) $(CFLAGS_CC_$(BUILDTYPE))')
            if self.flavor == 'mac':
                self.WriteLn('$(OBJS): GYP_OBJCFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) %s ' % precompiled_header.GetInclude('m') + '$(CFLAGS_$(BUILDTYPE)) $(CFLAGS_C_$(BUILDTYPE)) $(CFLAGS_OBJC_$(BUILDTYPE))')
                self.WriteLn('$(OBJS): GYP_OBJCXXFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) %s ' % precompiled_header.GetInclude('mm') + '$(CFLAGS_$(BUILDTYPE)) $(CFLAGS_CC_$(BUILDTYPE)) $(CFLAGS_OBJCC_$(BUILDTYPE))')
        self.WritePchTargets(precompiled_header.GetPchBuildCommands())
        extra_link_deps += [source for source in sources if Linkable(source)]
        self.WriteLn()

    def WritePchTargets(self, pch_commands):
        """Writes make rules to compile prefix headers."""
        if not pch_commands:
            return
        for gch, lang_flag, lang, input in pch_commands:
            extra_flags = {'c': '$(CFLAGS_C_$(BUILDTYPE))', 'cc': '$(CFLAGS_CC_$(BUILDTYPE))', 'm': '$(CFLAGS_C_$(BUILDTYPE)) $(CFLAGS_OBJC_$(BUILDTYPE))', 'mm': '$(CFLAGS_CC_$(BUILDTYPE)) $(CFLAGS_OBJCC_$(BUILDTYPE))'}[lang]
            var_name = {'c': 'GYP_PCH_CFLAGS', 'cc': 'GYP_PCH_CXXFLAGS', 'm': 'GYP_PCH_OBJCFLAGS', 'mm': 'GYP_PCH_OBJCXXFLAGS'}[lang]
            self.WriteLn(f'{gch}: {var_name} := {lang_flag} ' + '$(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) $(CFLAGS_$(BUILDTYPE)) ' + extra_flags)
            self.WriteLn(f'{gch}: {input} FORCE_DO_CMD')
            self.WriteLn('\t@$(call do_cmd,pch_%s,1)' % lang)
            self.WriteLn('')
            assert ' ' not in gch, 'Spaces in gch filenames not supported (%s)' % gch
            self.WriteLn('all_deps += %s' % gch)
            self.WriteLn('')

    def ComputeOutputBasename(self, spec):
        """Return the 'output basename' of a gyp spec.

        E.g., the loadable module 'foobar' in directory 'baz' will produce
          'libfoobar.so'
        """
        assert not self.is_mac_bundle
        if self.flavor == 'mac' and self.type in ('static_library', 'executable', 'shared_library', 'loadable_module'):
            return self.xcode_settings.GetExecutablePath()
        target = spec['target_name']
        target_prefix = ''
        target_ext = ''
        if self.type == 'static_library':
            if target[:3] == 'lib':
                target = target[3:]
            target_prefix = 'lib'
            target_ext = '.a'
        elif self.type in ('loadable_module', 'shared_library'):
            if target[:3] == 'lib':
                target = target[3:]
            target_prefix = 'lib'
            if self.flavor == 'aix':
                target_ext = '.a'
            else:
                target_ext = '.so'
        elif self.type == 'none':
            target = '%s.stamp' % target
        elif self.type != 'executable':
            print('ERROR: What output file should be generated?', 'type', self.type, 'target', target)
        target_prefix = spec.get('product_prefix', target_prefix)
        target = spec.get('product_name', target)
        product_ext = spec.get('product_extension')
        if product_ext:
            target_ext = '.' + product_ext
        return target_prefix + target + target_ext

    def _InstallImmediately(self):
        return self.toolset == 'target' and self.flavor == 'mac' and (self.type in ('static_library', 'executable', 'shared_library', 'loadable_module'))

    def ComputeOutput(self, spec):
        """Return the 'output' (full output path) of a gyp spec.

        E.g., the loadable module 'foobar' in directory 'baz' will produce
          '$(obj)/baz/libfoobar.so'
        """
        assert not self.is_mac_bundle
        path = os.path.join('$(obj).' + self.toolset, self.path)
        if self.type == 'executable' or self._InstallImmediately():
            path = '$(builddir)'
        path = spec.get('product_dir', path)
        return os.path.join(path, self.ComputeOutputBasename(spec))

    def ComputeMacBundleOutput(self, spec):
        """Return the 'output' (full output path) to a bundle output directory."""
        assert self.is_mac_bundle
        path = generator_default_variables['PRODUCT_DIR']
        return os.path.join(path, self.xcode_settings.GetWrapperName())

    def ComputeMacBundleBinaryOutput(self, spec):
        """Return the 'output' (full output path) to the binary in a bundle."""
        path = generator_default_variables['PRODUCT_DIR']
        return os.path.join(path, self.xcode_settings.GetExecutablePath())

    def ComputeDeps(self, spec):
        """Compute the dependencies of a gyp spec.

        Returns a tuple (deps, link_deps), where each is a list of
        filenames that will need to be put in front of make for either
        building (deps) or linking (link_deps).
        """
        deps = []
        link_deps = []
        if 'dependencies' in spec:
            deps.extend([target_outputs[dep] for dep in spec['dependencies'] if target_outputs[dep]])
            for dep in spec['dependencies']:
                if dep in target_link_deps:
                    link_deps.append(target_link_deps[dep])
            deps.extend(link_deps)
        return (gyp.common.uniquer(deps), gyp.common.uniquer(link_deps))

    def WriteDependencyOnExtraOutputs(self, target, extra_outputs):
        self.WriteMakeRule([self.output_binary], extra_outputs, comment='Build our special outputs first.', order_only=True)

    def WriteTarget(self, spec, configs, deps, link_deps, bundle_deps, extra_outputs, part_of_all):
        """Write Makefile code to produce the final target of the gyp spec.

        spec, configs: input from gyp.
        deps, link_deps: dependency lists; see ComputeDeps()
        extra_outputs: any extra outputs that our target should depend on
        part_of_all: flag indicating this target is part of 'all'
        """
        self.WriteLn('### Rules for final target.')
        if extra_outputs:
            self.WriteDependencyOnExtraOutputs(self.output_binary, extra_outputs)
            self.WriteMakeRule(extra_outputs, deps, comment='Preserve order dependency of special output on deps.', order_only=True)
        target_postbuilds = {}
        if self.type != 'none':
            for configname in sorted(configs.keys()):
                config = configs[configname]
                if self.flavor == 'mac':
                    ldflags = self.xcode_settings.GetLdflags(configname, generator_default_variables['PRODUCT_DIR'], lambda p: Sourceify(self.Absolutify(p)), arch=config.get('xcode_configuration_platform'))
                    gyp_to_build = gyp.common.InvertRelativePath(self.path)
                    target_postbuild = self.xcode_settings.AddImplicitPostbuilds(configname, QuoteSpaces(os.path.normpath(os.path.join(gyp_to_build, self.output))), QuoteSpaces(os.path.normpath(os.path.join(gyp_to_build, self.output_binary))))
                    if target_postbuild:
                        target_postbuilds[configname] = target_postbuild
                else:
                    ldflags = config.get('ldflags', [])
                    if any((dep.endswith('.so') or '.so.' in dep for dep in deps)):
                        ldflags.append('-Wl,-rpath=\\$$ORIGIN/')
                        ldflags.append('-Wl,-rpath-link=\\$(builddir)/')
                library_dirs = config.get('library_dirs', [])
                ldflags += ['-L%s' % library_dir for library_dir in library_dirs]
                self.WriteList(ldflags, 'LDFLAGS_%s' % configname)
                if self.flavor == 'mac':
                    self.WriteList(self.xcode_settings.GetLibtoolflags(configname), 'LIBTOOLFLAGS_%s' % configname)
            libraries = spec.get('libraries')
            if libraries:
                libraries = gyp.common.uniquer(libraries)
                if self.flavor == 'mac':
                    libraries = self.xcode_settings.AdjustLibraries(libraries)
            self.WriteList(libraries, 'LIBS')
            self.WriteLn('%s: GYP_LDFLAGS := $(LDFLAGS_$(BUILDTYPE))' % QuoteSpaces(self.output_binary))
            self.WriteLn('%s: LIBS := $(LIBS)' % QuoteSpaces(self.output_binary))
            if self.flavor == 'mac':
                self.WriteLn('%s: GYP_LIBTOOLFLAGS := $(LIBTOOLFLAGS_$(BUILDTYPE))' % QuoteSpaces(self.output_binary))
        postbuilds = []
        if self.flavor == 'mac':
            if target_postbuilds:
                postbuilds.append('$(TARGET_POSTBUILDS_$(BUILDTYPE))')
            postbuilds.extend(gyp.xcode_emulation.GetSpecPostbuildCommands(spec))
        if postbuilds:
            self.WriteSortedXcodeEnv(self.output, self.GetSortedXcodePostbuildEnv())
            for configname in target_postbuilds:
                self.WriteLn('%s: TARGET_POSTBUILDS_%s := %s' % (QuoteSpaces(self.output), configname, gyp.common.EncodePOSIXShellList(target_postbuilds[configname])))
            postbuilds.insert(0, gyp.common.EncodePOSIXShellList(['cd', self.path]))
            for i, postbuild in enumerate(postbuilds):
                if not postbuild.startswith('$'):
                    postbuilds[i] = EscapeShellArgument(postbuild)
            self.WriteLn('%s: builddir := $(abs_builddir)' % QuoteSpaces(self.output))
            self.WriteLn('%s: POSTBUILDS := %s' % (QuoteSpaces(self.output), ' '.join(postbuilds)))
        if self.is_mac_bundle:
            self.WriteDependencyOnExtraOutputs(self.output, extra_outputs)
            self.WriteList([QuoteSpaces(dep) for dep in bundle_deps], 'BUNDLE_DEPS')
            self.WriteLn('%s: $(BUNDLE_DEPS)' % QuoteSpaces(self.output))
            if self.type in ('shared_library', 'loadable_module'):
                self.WriteLn('\t@$(call do_cmd,mac_package_framework,,,%s)' % self.xcode_settings.GetFrameworkVersion())
            if postbuilds:
                self.WriteLn('\t@$(call do_postbuilds)')
            postbuilds = []
            self.WriteLn('\t@true  # No-op, used by tests')
            self.WriteLn('\t@touch -c %s' % QuoteSpaces(self.output))
        if postbuilds:
            assert not self.is_mac_bundle, "Postbuilds for bundles should be done on the bundle, not the binary (target '%s')" % self.target
            assert 'product_dir' not in spec, 'Postbuilds do not work with custom product_dir'
        if self.type == 'executable':
            self.WriteLn('%s: LD_INPUTS := %s' % (QuoteSpaces(self.output_binary), ' '.join((QuoteSpaces(dep) for dep in link_deps))))
            if self.toolset == 'host' and self.flavor == 'android':
                self.WriteDoCmd([self.output_binary], link_deps, 'link_host', part_of_all, postbuilds=postbuilds)
            else:
                self.WriteDoCmd([self.output_binary], link_deps, 'link', part_of_all, postbuilds=postbuilds)
        elif self.type == 'static_library':
            for link_dep in link_deps:
                assert ' ' not in link_dep, 'Spaces in alink input filenames not supported (%s)' % link_dep
            if self.flavor not in ('mac', 'openbsd', 'netbsd', 'win') and (not self.is_standalone_static_library):
                self.WriteDoCmd([self.output_binary], link_deps, 'alink_thin', part_of_all, postbuilds=postbuilds)
            else:
                self.WriteDoCmd([self.output_binary], link_deps, 'alink', part_of_all, postbuilds=postbuilds)
        elif self.type == 'shared_library':
            self.WriteLn('%s: LD_INPUTS := %s' % (QuoteSpaces(self.output_binary), ' '.join((QuoteSpaces(dep) for dep in link_deps))))
            self.WriteDoCmd([self.output_binary], link_deps, 'solink', part_of_all, postbuilds=postbuilds)
        elif self.type == 'loadable_module':
            for link_dep in link_deps:
                assert ' ' not in link_dep, 'Spaces in module input filenames not supported (%s)' % link_dep
            if self.toolset == 'host' and self.flavor == 'android':
                self.WriteDoCmd([self.output_binary], link_deps, 'solink_module_host', part_of_all, postbuilds=postbuilds)
            else:
                self.WriteDoCmd([self.output_binary], link_deps, 'solink_module', part_of_all, postbuilds=postbuilds)
        elif self.type == 'none':
            self.WriteDoCmd([self.output_binary], deps, 'touch', part_of_all, postbuilds=postbuilds)
        else:
            print('WARNING: no output for', self.type, self.target)
        if (self.output and self.output != self.target) and self.type not in self._INSTALLABLE_TARGETS:
            self.WriteMakeRule([self.target], [self.output], comment='Add target alias', phony=True)
            if part_of_all:
                self.WriteMakeRule(['all'], [self.target], comment='Add target alias to "all" target.', phony=True)
        if self.type in self._INSTALLABLE_TARGETS or self.is_standalone_static_library:
            if self.type == 'shared_library':
                file_desc = 'shared library'
            elif self.type == 'static_library':
                file_desc = 'static library'
            else:
                file_desc = 'executable'
            install_path = self._InstallableTargetInstallPath()
            installable_deps = [self.output]
            if self.flavor == 'mac' and 'product_dir' not in spec and (self.toolset == 'target'):
                assert install_path == self.output, '{} != {}'.format(install_path, self.output)
            self.WriteMakeRule([self.target], [install_path], comment='Add target alias', phony=True)
            if install_path != self.output:
                assert not self.is_mac_bundle
                self.WriteDoCmd([install_path], [self.output], 'copy', comment='Copy this to the %s output path.' % file_desc, part_of_all=part_of_all)
                installable_deps.append(install_path)
            if self.output != self.alias and self.alias != self.target:
                self.WriteMakeRule([self.alias], installable_deps, comment='Short alias for building this %s.' % file_desc, phony=True)
            if part_of_all:
                self.WriteMakeRule(['all'], [install_path], comment='Add %s to "all" target.' % file_desc, phony=True)

    def WriteList(self, value_list, variable=None, prefix='', quoter=QuoteIfNecessary):
        """Write a variable definition that is a list of values.

        E.g. WriteList(['a','b'], 'foo', prefix='blah') writes out
             foo = blaha blahb
        but in a pretty-printed style.
        """
        values = ''
        if value_list:
            value_list = [quoter(prefix + value) for value in value_list]
            values = ' \\\n\t' + ' \\\n\t'.join(value_list)
        self.fp.write(f'{variable} :={values}\n\n')

    def WriteDoCmd(self, outputs, inputs, command, part_of_all, comment=None, postbuilds=False):
        """Write a Makefile rule that uses do_cmd.

        This makes the outputs dependent on the command line that was run,
        as well as support the V= make command line flag.
        """
        suffix = ''
        if postbuilds:
            assert ',' not in command
            suffix = ',,1'
        self.WriteMakeRule(outputs, inputs, actions=[f'$(call do_cmd,{command}{suffix})'], comment=comment, command=command, force=True)
        outputs = [QuoteSpaces(o, SPACE_REPLACEMENT) for o in outputs]
        self.WriteLn('all_deps += %s' % ' '.join(outputs))

    def WriteMakeRule(self, outputs, inputs, actions=None, comment=None, order_only=False, force=False, phony=False, command=None):
        """Write a Makefile rule, with some extra tricks.

        outputs: a list of outputs for the rule (note: this is not directly
                 supported by make; see comments below)
        inputs: a list of inputs for the rule
        actions: a list of shell commands to run for the rule
        comment: a comment to put in the Makefile above the rule (also useful
                 for making this Python script's code self-documenting)
        order_only: if true, makes the dependency order-only
        force: if true, include FORCE_DO_CMD as an order-only dep
        phony: if true, the rule does not actually generate the named output, the
               output is just a name to run the rule
        command: (optional) command name to generate unambiguous labels
        """
        outputs = [QuoteSpaces(o) for o in outputs]
        inputs = [QuoteSpaces(i) for i in inputs]
        if comment:
            self.WriteLn('# ' + comment)
        if phony:
            self.WriteLn('.PHONY: ' + ' '.join(outputs))
        if actions:
            self.WriteLn('%s: TOOLSET := $(TOOLSET)' % outputs[0])
        force_append = ' FORCE_DO_CMD' if force else ''
        if order_only:
            self.WriteLn('{}: | {}{}'.format(' '.join(outputs), ' '.join(inputs), force_append))
        elif len(outputs) == 1:
            self.WriteLn('{}: {}{}'.format(outputs[0], ' '.join(inputs), force_append))
        else:
            cmddigest = hashlib.sha1((command or self.target).encode('utf-8')).hexdigest()
            intermediate = '%s.intermediate' % cmddigest
            self.WriteLn('{}: {}'.format(' '.join(outputs), intermediate))
            self.WriteLn('\t%s' % '@:')
            self.WriteLn('{}: {}'.format('.INTERMEDIATE', intermediate))
            self.WriteLn('{}: {}{}'.format(intermediate, ' '.join(inputs), force_append))
            actions.insert(0, '$(call do_cmd,touch)')
        if actions:
            for action in actions:
                self.WriteLn('\t%s' % action)
        self.WriteLn()

    def WriteAndroidNdkModuleRule(self, module_name, all_sources, link_deps):
        """Write a set of LOCAL_XXX definitions for Android NDK.

        These variable definitions will be used by Android NDK but do nothing for
        non-Android applications.

        Arguments:
          module_name: Android NDK module name, which must be unique among all
              module names.
          all_sources: A list of source files (will be filtered by Compilable).
          link_deps: A list of link dependencies, which must be sorted in
              the order from dependencies to dependents.
        """
        if self.type not in ('executable', 'shared_library', 'static_library'):
            return
        self.WriteLn('# Variable definitions for Android applications')
        self.WriteLn('include $(CLEAR_VARS)')
        self.WriteLn('LOCAL_MODULE := ' + module_name)
        self.WriteLn('LOCAL_CFLAGS := $(CFLAGS_$(BUILDTYPE)) $(DEFS_$(BUILDTYPE)) $(CFLAGS_C_$(BUILDTYPE)) $(INCS_$(BUILDTYPE))')
        self.WriteLn('LOCAL_CPPFLAGS := $(CFLAGS_CC_$(BUILDTYPE))')
        self.WriteLn('LOCAL_C_INCLUDES :=')
        self.WriteLn('LOCAL_LDLIBS := $(LDFLAGS_$(BUILDTYPE)) $(LIBS)')
        cpp_ext = {'.cc': 0, '.cpp': 0, '.cxx': 0}
        default_cpp_ext = '.cpp'
        for filename in all_sources:
            ext = os.path.splitext(filename)[1]
            if ext in cpp_ext:
                cpp_ext[ext] += 1
                if cpp_ext[ext] > cpp_ext[default_cpp_ext]:
                    default_cpp_ext = ext
        self.WriteLn('LOCAL_CPP_EXTENSION := ' + default_cpp_ext)
        self.WriteList(list(map(self.Absolutify, filter(Compilable, all_sources))), 'LOCAL_SRC_FILES')

        def DepsToModules(deps, prefix, suffix):
            modules = []
            for filepath in deps:
                filename = os.path.basename(filepath)
                if filename.startswith(prefix) and filename.endswith(suffix):
                    modules.append(filename[len(prefix):-len(suffix)])
            return modules
        params = {'flavor': 'linux'}
        default_variables = {}
        CalculateVariables(default_variables, params)
        self.WriteList(DepsToModules(link_deps, generator_default_variables['SHARED_LIB_PREFIX'], default_variables['SHARED_LIB_SUFFIX']), 'LOCAL_SHARED_LIBRARIES')
        self.WriteList(DepsToModules(link_deps, generator_default_variables['STATIC_LIB_PREFIX'], generator_default_variables['STATIC_LIB_SUFFIX']), 'LOCAL_STATIC_LIBRARIES')
        if self.type == 'executable':
            self.WriteLn('include $(BUILD_EXECUTABLE)')
        elif self.type == 'shared_library':
            self.WriteLn('include $(BUILD_SHARED_LIBRARY)')
        elif self.type == 'static_library':
            self.WriteLn('include $(BUILD_STATIC_LIBRARY)')
        self.WriteLn()

    def WriteLn(self, text=''):
        self.fp.write(text + '\n')

    def GetSortedXcodeEnv(self, additional_settings=None):
        return gyp.xcode_emulation.GetSortedXcodeEnv(self.xcode_settings, '$(abs_builddir)', os.path.join('$(abs_srcdir)', self.path), '$(BUILDTYPE)', additional_settings)

    def GetSortedXcodePostbuildEnv(self):
        strip_save_file = self.xcode_settings.GetPerTargetSetting('CHROMIUM_STRIP_SAVE_FILE', '')
        return self.GetSortedXcodeEnv(additional_settings={'CHROMIUM_STRIP_SAVE_FILE': strip_save_file})

    def WriteSortedXcodeEnv(self, target, env):
        for k, v in env:
            self.WriteLn(f'{QuoteSpaces(target)}: export {k} := {v}')

    def Objectify(self, path):
        """Convert a path to its output directory form."""
        if '$(' in path:
            path = path.replace('$(obj)/', '$(obj).%s/$(TARGET)/' % self.toolset)
        if '$(obj)' not in path:
            path = f'$(obj).{self.toolset}/$(TARGET)/{path}'
        return path

    def Pchify(self, path, lang):
        """Convert a prefix header path to its output directory form."""
        path = self.Absolutify(path)
        if '$(' in path:
            path = path.replace('$(obj)/', f'$(obj).{self.toolset}/$(TARGET)/pch-{lang}')
            return path
        return f'$(obj).{self.toolset}/$(TARGET)/pch-{lang}/{path}'

    def Absolutify(self, path):
        """Convert a subdirectory-relative path into a base-relative path.
        Skips over paths that contain variables."""
        if '$(' in path:
            return path.rstrip('/')
        return os.path.normpath(os.path.join(self.path, path))

    def ExpandInputRoot(self, template, expansion, dirname):
        if '%(INPUT_ROOT)s' not in template and '%(INPUT_DIRNAME)s' not in template:
            return template
        path = template % {'INPUT_ROOT': expansion, 'INPUT_DIRNAME': dirname}
        return path

    def _InstallableTargetInstallPath(self):
        """Returns the location of the final output for an installable target."""
        return '$(builddir)/' + self.alias