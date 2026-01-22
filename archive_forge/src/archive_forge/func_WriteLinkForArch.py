import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def WriteLinkForArch(self, ninja_file, spec, config_name, config, link_deps, compile_deps, arch=None):
    """Write out a link step. Fills out target.binary. """
    command = {'executable': 'link', 'loadable_module': 'solink_module', 'shared_library': 'solink'}[spec['type']]
    command_suffix = ''
    implicit_deps = set()
    solibs = set()
    order_deps = set()
    if compile_deps:
        order_deps.add(compile_deps)
    if 'dependencies' in spec:
        extra_link_deps = set()
        for dep in spec['dependencies']:
            target = self.target_outputs.get(dep)
            if not target:
                continue
            linkable = target.Linkable()
            if linkable:
                new_deps = []
                if self.flavor == 'win' and target.component_objs and self.msvs_settings.IsUseLibraryDependencyInputs(config_name):
                    new_deps = target.component_objs
                    if target.compile_deps:
                        order_deps.add(target.compile_deps)
                elif self.flavor == 'win' and target.import_lib:
                    new_deps = [target.import_lib]
                elif target.UsesToc(self.flavor):
                    solibs.add(target.binary)
                    implicit_deps.add(target.binary + '.TOC')
                else:
                    new_deps = [target.binary]
                for new_dep in new_deps:
                    if new_dep not in extra_link_deps:
                        extra_link_deps.add(new_dep)
                        link_deps.append(new_dep)
            final_output = target.FinalOutput()
            if not linkable or final_output != target.binary:
                implicit_deps.add(final_output)
    extra_bindings = []
    if self.target.uses_cpp and self.flavor != 'win':
        extra_bindings.append(('ld', '$ldxx'))
    output = self.ComputeOutput(spec, arch)
    if arch is None and (not self.is_mac_bundle):
        self.AppendPostbuildVariable(extra_bindings, spec, output, output)
    is_executable = spec['type'] == 'executable'
    if self.toolset == 'target':
        env_ldflags = os.environ.get('LDFLAGS', '').split()
    elif self.toolset == 'host':
        env_ldflags = os.environ.get('LDFLAGS_host', '').split()
    if self.flavor == 'mac':
        ldflags = self.xcode_settings.GetLdflags(config_name, self.ExpandSpecial(generator_default_variables['PRODUCT_DIR']), self.GypPathToNinja, arch)
        ldflags = env_ldflags + ldflags
    elif self.flavor == 'win':
        manifest_base_name = self.GypPathToUniqueOutput(self.ComputeOutputFileName(spec))
        ldflags, intermediate_manifest, manifest_files = self.msvs_settings.GetLdflags(config_name, self.GypPathToNinja, self.ExpandSpecial, manifest_base_name, output, is_executable, self.toplevel_build)
        ldflags = env_ldflags + ldflags
        self.WriteVariableList(ninja_file, 'manifests', manifest_files)
        implicit_deps = implicit_deps.union(manifest_files)
        if intermediate_manifest:
            self.WriteVariableList(ninja_file, 'intermediatemanifest', [intermediate_manifest])
        command_suffix = _GetWinLinkRuleNameSuffix(self.msvs_settings.IsEmbedManifest(config_name))
        def_file = self.msvs_settings.GetDefFile(self.GypPathToNinja)
        if def_file:
            implicit_deps.add(def_file)
    else:
        ldflags = env_ldflags + config.get('ldflags', [])
        if is_executable and len(solibs):
            rpath = 'lib/'
            if self.toolset != 'target':
                rpath += self.toolset
                ldflags.append('-Wl,-rpath=\\$$ORIGIN/%s' % rpath)
            else:
                ldflags.append('-Wl,-rpath=%s' % self.target_rpath)
            ldflags.append('-Wl,-rpath-link=%s' % rpath)
    self.WriteVariableList(ninja_file, 'ldflags', map(self.ExpandSpecial, ldflags))
    library_dirs = config.get('library_dirs', [])
    if self.flavor == 'win':
        library_dirs = [self.msvs_settings.ConvertVSMacros(library_dir, config_name) for library_dir in library_dirs]
        library_dirs = ['/LIBPATH:' + QuoteShellArgument(self.GypPathToNinja(library_dir), self.flavor) for library_dir in library_dirs]
    else:
        library_dirs = [QuoteShellArgument('-L' + self.GypPathToNinja(library_dir), self.flavor) for library_dir in library_dirs]
    libraries = gyp.common.uniquer(map(self.ExpandSpecial, spec.get('libraries', [])))
    if self.flavor == 'mac':
        libraries = self.xcode_settings.AdjustLibraries(libraries, config_name)
    elif self.flavor == 'win':
        libraries = self.msvs_settings.AdjustLibraries(libraries)
    self.WriteVariableList(ninja_file, 'libs', library_dirs + libraries)
    linked_binary = output
    if command in ('solink', 'solink_module'):
        extra_bindings.append(('soname', os.path.split(output)[1]))
        extra_bindings.append(('lib', gyp.common.EncodePOSIXShellArgument(output)))
        if self.flavor != 'win':
            link_file_list = output
            if self.is_mac_bundle:
                link_file_list = self.xcode_settings.GetWrapperName()
            if arch:
                link_file_list += '.' + arch
            link_file_list += '.rsp'
            link_file_list = link_file_list.replace(' ', '_')
            extra_bindings.append(('link_file_list', gyp.common.EncodePOSIXShellArgument(link_file_list)))
        if self.flavor == 'win':
            extra_bindings.append(('binary', output))
            if '/NOENTRY' not in ldflags and (not self.msvs_settings.GetNoImportLibrary(config_name)):
                self.target.import_lib = output + '.lib'
                extra_bindings.append(('implibflag', '/IMPLIB:%s' % self.target.import_lib))
                pdbname = self.msvs_settings.GetPDBName(config_name, self.ExpandSpecial, output + '.pdb')
                output = [output, self.target.import_lib]
                if pdbname:
                    output.append(pdbname)
        elif not self.is_mac_bundle:
            output = [output, output + '.TOC']
        else:
            command = command + '_notoc'
    elif self.flavor == 'win':
        extra_bindings.append(('binary', output))
        pdbname = self.msvs_settings.GetPDBName(config_name, self.ExpandSpecial, output + '.pdb')
        if pdbname:
            output = [output, pdbname]
    if len(solibs):
        extra_bindings.append(('solibs', gyp.common.EncodePOSIXShellList(sorted(solibs))))
    ninja_file.build(output, command + command_suffix, link_deps, implicit=sorted(implicit_deps), order_only=list(order_deps), variables=extra_bindings)
    return linked_binary