import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
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