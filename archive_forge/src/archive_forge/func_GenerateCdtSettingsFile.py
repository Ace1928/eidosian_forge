from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def GenerateCdtSettingsFile(target_list, target_dicts, data, params, config_name, out_name, options, shared_intermediate_dirs):
    gyp.common.EnsureDirExists(out_name)
    with open(out_name, 'w') as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write('<cdtprojectproperties>\n')
        eclipse_langs = ['C++ Source File', 'C Source File', 'Assembly Source File', 'GNU C++', 'GNU C', 'Assembly']
        compiler_path = GetCompilerPath(target_list, data, options)
        include_dirs = GetAllIncludeDirectories(target_list, target_dicts, shared_intermediate_dirs, config_name, params, compiler_path)
        WriteIncludePaths(out, eclipse_langs, include_dirs)
        defines = GetAllDefines(target_list, target_dicts, data, config_name, params, compiler_path)
        WriteMacros(out, eclipse_langs, defines)
        out.write('</cdtprojectproperties>\n')