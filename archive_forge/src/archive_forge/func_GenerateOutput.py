import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def GenerateOutput(target_list, target_dicts, data, params):
    options = params['options']
    flavor = gyp.common.GetFlavor(params)
    generator_flags = params.get('generator_flags', {})
    builddir_name = generator_flags.get('output_dir', 'out')
    android_ndk_version = generator_flags.get('android_ndk_version', None)
    default_target = generator_flags.get('default_target', 'all')

    def CalculateMakefilePath(build_file, base_name):
        """Determine where to write a Makefile for a given gyp file."""
        base_path = gyp.common.RelativePath(os.path.dirname(build_file), options.depth)
        output_file = os.path.join(options.depth, base_path, base_name)
        if options.generator_output:
            output_file = os.path.join(options.depth, options.generator_output, base_path, base_name)
        base_path = gyp.common.RelativePath(os.path.dirname(build_file), options.toplevel_dir)
        return (base_path, output_file)
    default_configuration = None
    toolsets = {target_dicts[target]['toolset'] for target in target_list}
    for target in target_list:
        spec = target_dicts[target]
        if spec['default_configuration'] != 'Default':
            default_configuration = spec['default_configuration']
            break
    if not default_configuration:
        default_configuration = 'Default'
    srcdir = '.'
    makefile_name = 'Makefile' + options.suffix
    makefile_path = os.path.join(options.toplevel_dir, makefile_name)
    if options.generator_output:
        global srcdir_prefix
        makefile_path = os.path.join(options.toplevel_dir, options.generator_output, makefile_name)
        srcdir = gyp.common.RelativePath(srcdir, options.generator_output)
        srcdir_prefix = '$(srcdir)/'
    flock_command = 'flock'
    copy_archive_arguments = '-af'
    makedep_arguments = '-MMD'
    header_params = {'default_target': default_target, 'builddir': builddir_name, 'default_configuration': default_configuration, 'flock': flock_command, 'flock_index': 1, 'link_commands': LINK_COMMANDS_LINUX, 'extra_commands': '', 'srcdir': srcdir, 'copy_archive_args': copy_archive_arguments, 'makedep_args': makedep_arguments, 'CC.target': GetEnvironFallback(('CC_target', 'CC'), '$(CC)'), 'AR.target': GetEnvironFallback(('AR_target', 'AR'), '$(AR)'), 'CXX.target': GetEnvironFallback(('CXX_target', 'CXX'), '$(CXX)'), 'LINK.target': GetEnvironFallback(('LINK_target', 'LINK'), '$(LINK)'), 'CC.host': GetEnvironFallback(('CC_host', 'CC'), 'gcc'), 'AR.host': GetEnvironFallback(('AR_host', 'AR'), 'ar'), 'CXX.host': GetEnvironFallback(('CXX_host', 'CXX'), 'g++'), 'LINK.host': GetEnvironFallback(('LINK_host', 'LINK'), '$(CXX.host)')}
    if flavor == 'mac':
        flock_command = './gyp-mac-tool flock'
        header_params.update({'flock': flock_command, 'flock_index': 2, 'link_commands': LINK_COMMANDS_MAC, 'extra_commands': SHARED_HEADER_MAC_COMMANDS})
    elif flavor == 'android':
        header_params.update({'link_commands': LINK_COMMANDS_ANDROID})
    elif flavor == 'zos':
        copy_archive_arguments = '-fPR'
        makedep_arguments = '-qmakedep=gcc'
        header_params.update({'copy_archive_args': copy_archive_arguments, 'makedep_args': makedep_arguments, 'link_commands': LINK_COMMANDS_OS390, 'CC.target': GetEnvironFallback(('CC_target', 'CC'), 'njsc'), 'CXX.target': GetEnvironFallback(('CXX_target', 'CXX'), 'njsc++'), 'CC.host': GetEnvironFallback(('CC_host', 'CC'), 'njsc'), 'CXX.host': GetEnvironFallback(('CXX_host', 'CXX'), 'njsc++')})
    elif flavor == 'solaris':
        copy_archive_arguments = '-pPRf@'
        header_params.update({'copy_archive_args': copy_archive_arguments, 'flock': './gyp-flock-tool flock', 'flock_index': 2})
    elif flavor == 'freebsd':
        header_params.update({'flock': 'lockf'})
    elif flavor == 'openbsd':
        copy_archive_arguments = '-pPRf'
        header_params.update({'copy_archive_args': copy_archive_arguments})
    elif flavor == 'aix':
        copy_archive_arguments = '-pPRf'
        header_params.update({'copy_archive_args': copy_archive_arguments, 'link_commands': LINK_COMMANDS_AIX, 'flock': './gyp-flock-tool flock', 'flock_index': 2})
    build_file, _, _ = gyp.common.ParseQualifiedTarget(target_list[0])
    make_global_settings_array = data[build_file].get('make_global_settings', [])
    wrappers = {}
    for key, value in make_global_settings_array:
        if key.endswith('_wrapper'):
            wrappers[key[:-len('_wrapper')]] = '$(abspath %s)' % value
    make_global_settings = ''
    for key, value in make_global_settings_array:
        if re.match('.*_wrapper', key):
            continue
        if value[0] != '$':
            value = '$(abspath %s)' % value
        wrapper = wrappers.get(key)
        if wrapper:
            value = f'{wrapper} {value}'
            del wrappers[key]
        if key in ('CC', 'CC.host', 'CXX', 'CXX.host'):
            make_global_settings += 'ifneq (,$(filter $(origin %s), undefined default))\n' % key
            env_key = key.replace('.', '_')
            if env_key in os.environ:
                value = os.environ[env_key]
            make_global_settings += f'  {key} = {value}\n'
            make_global_settings += 'endif\n'
        else:
            make_global_settings += f'{key} ?= {value}\n'
    header_params['make_global_settings'] = make_global_settings
    gyp.common.EnsureDirExists(makefile_path)
    root_makefile = open(makefile_path, 'w')
    root_makefile.write(SHARED_HEADER % header_params)
    if android_ndk_version:
        root_makefile.write('# Define LOCAL_PATH for build of Android applications.\nLOCAL_PATH := $(call my-dir)\n\n')
    for toolset in toolsets:
        root_makefile.write('TOOLSET := %s\n' % toolset)
        WriteRootHeaderSuffixRules(root_makefile)
    dest_path = os.path.dirname(makefile_path)
    gyp.common.CopyTool(flavor, dest_path)
    needed_targets = set()
    for build_file in params['build_files']:
        for target in gyp.common.AllTargets(target_list, target_dicts, build_file):
            needed_targets.add(target)
    build_files = set()
    include_list = set()
    for qualified_target in target_list:
        build_file, target, toolset = gyp.common.ParseQualifiedTarget(qualified_target)
        this_make_global_settings = data[build_file].get('make_global_settings', [])
        assert make_global_settings_array == this_make_global_settings, f'make_global_settings needs to be the same for all targets {this_make_global_settings} vs. {make_global_settings}'
        build_files.add(gyp.common.RelativePath(build_file, options.toplevel_dir))
        included_files = data[build_file]['included_files']
        for included_file in included_files:
            relative_include_file = gyp.common.RelativePath(gyp.common.UnrelativePath(included_file, build_file), options.toplevel_dir)
            abs_include_file = os.path.abspath(relative_include_file)
            if params['home_dot_gyp'] and abs_include_file.startswith(params['home_dot_gyp']):
                build_files.add(abs_include_file)
            else:
                build_files.add(relative_include_file)
        base_path, output_file = CalculateMakefilePath(build_file, target + '.' + toolset + options.suffix + '.mk')
        spec = target_dicts[qualified_target]
        configs = spec['configurations']
        if flavor == 'mac':
            gyp.xcode_emulation.MergeGlobalXcodeSettingsToSpec(data[build_file], spec)
        writer = MakefileWriter(generator_flags, flavor)
        writer.Write(qualified_target, base_path, output_file, spec, configs, part_of_all=qualified_target in needed_targets)
        mkfile_rel_path = gyp.common.RelativePath(output_file, os.path.dirname(makefile_path))
        include_list.add(mkfile_rel_path)
    depth_rel_path = gyp.common.RelativePath(options.depth, os.getcwd())
    for build_file in build_files:
        build_file = os.path.join(depth_rel_path, build_file)
        gyp_targets = [target_dicts[qualified_target]['target_name'] for qualified_target in target_list if qualified_target.startswith(build_file) and qualified_target in needed_targets]
        if not gyp_targets:
            continue
        base_path, output_file = CalculateMakefilePath(build_file, os.path.splitext(os.path.basename(build_file))[0] + '.Makefile')
        makefile_rel_path = gyp.common.RelativePath(os.path.dirname(makefile_path), os.path.dirname(output_file))
        writer.WriteSubMake(output_file, makefile_rel_path, gyp_targets, builddir_name)
    root_makefile.write('\n')
    for include_file in sorted(include_list):
        root_makefile.write('ifeq ($(strip $(foreach prefix,$(NO_LOAD),\\\n    $(findstring $(join ^,$(prefix)),\\\n                 $(join ^,' + include_file + ')))),)\n')
        root_makefile.write('  include ' + include_file + '\n')
        root_makefile.write('endif\n')
    root_makefile.write('\n')
    if not generator_flags.get('standalone') and generator_flags.get('auto_regeneration', True):
        WriteAutoRegenerationRule(params, root_makefile, makefile_name, build_files)
    root_makefile.write(SHARED_FOOTER)
    root_makefile.close()