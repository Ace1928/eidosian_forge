from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def GetAllDefines(target_list, target_dicts, data, config_name, params, compiler_path):
    """Calculate the defines for a project.

  Returns:
    A dict that includes explicit defines declared in gyp files along with all
    of the default defines that the compiler uses.
  """
    all_defines = {}
    flavor = gyp.common.GetFlavor(params)
    if flavor == 'win':
        generator_flags = params.get('generator_flags', {})
    for target_name in target_list:
        target = target_dicts[target_name]
        if flavor == 'win':
            msvs_settings = gyp.msvs_emulation.MsvsSettings(target, generator_flags)
            extra_defines = msvs_settings.GetComputedDefines(config_name)
        else:
            extra_defines = []
        if config_name in target['configurations']:
            config = target['configurations'][config_name]
            target_defines = config['defines']
        else:
            target_defines = []
        for define in target_defines + extra_defines:
            split_define = define.split('=', 1)
            if len(split_define) == 1:
                split_define.append('1')
            if split_define[0].strip() in all_defines:
                continue
            all_defines[split_define[0].strip()] = split_define[1].strip()
    if flavor == 'win':
        return all_defines
    if compiler_path:
        command = shlex.split(compiler_path)
        command.extend(['-E', '-dM', '-'])
        cpp_proc = subprocess.Popen(args=command, cwd='.', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        cpp_output = cpp_proc.communicate()[0].decode('utf-8')
        cpp_lines = cpp_output.split('\n')
        for cpp_line in cpp_lines:
            if not cpp_line.strip():
                continue
            cpp_line_parts = cpp_line.split(' ', 2)
            key = cpp_line_parts[1]
            if len(cpp_line_parts) >= 3:
                val = cpp_line_parts[2]
            else:
                val = '1'
            all_defines[key] = val
    return all_defines