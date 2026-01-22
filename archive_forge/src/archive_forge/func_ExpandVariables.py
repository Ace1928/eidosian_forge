import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def ExpandVariables(input, phase, variables, build_file):
    if phase == PHASE_EARLY:
        variable_re = early_variable_re
        expansion_symbol = '<'
    elif phase == PHASE_LATE:
        variable_re = late_variable_re
        expansion_symbol = '>'
    elif phase == PHASE_LATELATE:
        variable_re = latelate_variable_re
        expansion_symbol = '^'
    else:
        assert False
    input_str = str(input)
    if IsStrCanonicalInt(input_str):
        return int(input_str)
    if expansion_symbol not in input_str:
        return input_str
    matches = list(variable_re.finditer(input_str))
    if not matches:
        return input_str
    output = input_str
    matches.reverse()
    for match_group in matches:
        match = match_group.groupdict()
        gyp.DebugOutput(gyp.DEBUG_VARIABLES, 'Matches: %r', match)
        run_command = '!' in match['type']
        command_string = match['command_string']
        file_list = '|' in match['type']
        replace_start = match_group.start('replace')
        replace_end = match_group.end('replace')
        c_start, c_end = FindEnclosingBracketGroup(input_str[replace_start:])
        replace_end = replace_start + c_end
        replacement = input_str[replace_start:replace_end]
        contents_start = replace_start + c_start + 1
        contents_end = replace_end - 1
        contents = input_str[contents_start:contents_end]
        if file_list:
            processed_variables = gyp.simple_copy.deepcopy(variables)
            ProcessListFiltersInDict(contents, processed_variables)
            contents = ExpandVariables(contents, phase, processed_variables, build_file)
        else:
            contents = ExpandVariables(contents, phase, variables, build_file)
        contents = contents.strip()
        expand_to_list = '@' in match['type'] and input_str == replacement
        if run_command or file_list:
            build_file_dir = os.path.dirname(build_file)
            if build_file_dir == '' and (not file_list):
                build_file_dir = None
        if file_list:
            if type(contents) is list:
                contents_list = contents
            else:
                contents_list = contents.split(' ')
            replacement = contents_list[0]
            if os.path.isabs(replacement):
                raise GypError('| cannot handle absolute paths, got "%s"' % replacement)
            if not generator_filelist_paths:
                path = os.path.join(build_file_dir, replacement)
            else:
                if os.path.isabs(build_file_dir):
                    toplevel = generator_filelist_paths['toplevel']
                    rel_build_file_dir = gyp.common.RelativePath(build_file_dir, toplevel)
                else:
                    rel_build_file_dir = build_file_dir
                qualified_out_dir = generator_filelist_paths['qualified_out_dir']
                path = os.path.join(qualified_out_dir, rel_build_file_dir, replacement)
                gyp.common.EnsureDirExists(path)
            replacement = gyp.common.RelativePath(path, build_file_dir)
            f = gyp.common.WriteOnDiff(path)
            for i in contents_list[1:]:
                f.write('%s\n' % i)
            f.close()
        elif run_command:
            use_shell = True
            if match['is_array']:
                contents = eval(contents)
                use_shell = False
            cache_key = (str(contents), build_file_dir)
            cached_value = cached_command_results.get(cache_key, None)
            if cached_value is None:
                gyp.DebugOutput(gyp.DEBUG_VARIABLES, "Executing command '%s' in directory '%s'", contents, build_file_dir)
                replacement = ''
                if command_string == 'pymod_do_main':
                    oldwd = os.getcwd()
                    if build_file_dir:
                        os.chdir(build_file_dir)
                    sys.path.append(os.getcwd())
                    try:
                        parsed_contents = shlex.split(contents)
                        try:
                            py_module = __import__(parsed_contents[0])
                        except ImportError as e:
                            raise GypError('Error importing pymod_do_mainmodule (%s): %s' % (parsed_contents[0], e))
                        replacement = str(py_module.DoMain(parsed_contents[1:])).rstrip()
                    finally:
                        sys.path.pop()
                        os.chdir(oldwd)
                    assert replacement is not None
                elif command_string:
                    raise GypError("Unknown command string '%s' in '%s'." % (command_string, contents))
                else:
                    contents = FixupPlatformCommand(contents)
                    try:
                        p = subprocess.Popen(contents, shell=use_shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, cwd=build_file_dir)
                    except Exception as e:
                        raise GypError("%s while executing command '%s' in %s" % (e, contents, build_file))
                    p_stdout, p_stderr = p.communicate('')
                    p_stdout = p_stdout.decode('utf-8')
                    p_stderr = p_stderr.decode('utf-8')
                    if p.wait() != 0 or p_stderr:
                        sys.stderr.write(p_stderr)
                        raise GypError("Call to '%s' returned exit status %d while in %s." % (contents, p.returncode, build_file))
                    replacement = p_stdout.rstrip()
                cached_command_results[cache_key] = replacement
            else:
                gyp.DebugOutput(gyp.DEBUG_VARIABLES, "Had cache value for command '%s' in directory '%s'", contents, build_file_dir)
                replacement = cached_value
        elif contents not in variables:
            if contents[-1] in ['!', '/']:
                replacement = []
            else:
                raise GypError('Undefined variable ' + contents + ' in ' + build_file)
        else:
            replacement = variables[contents]
        if isinstance(replacement, bytes) and (not isinstance(replacement, str)):
            replacement = replacement.decode('utf-8')
        if type(replacement) is list:
            for item in replacement:
                if isinstance(item, bytes) and (not isinstance(item, str)):
                    item = item.decode('utf-8')
                if not contents[-1] == '/' and type(item) not in (str, int):
                    raise GypError('Variable ' + contents + ' must expand to a string or list of strings; ' + 'list contains a ' + item.__class__.__name__)
            ProcessVariablesAndConditionsInList(replacement, phase, variables, build_file)
        elif type(replacement) not in (str, int):
            raise GypError('Variable ' + contents + ' must expand to a string or list of strings; ' + 'found a ' + replacement.__class__.__name__)
        if expand_to_list:
            if type(replacement) is list:
                output = replacement[:]
            else:
                output = shlex.split(str(replacement))
        else:
            encoded_replacement = ''
            if type(replacement) is list:
                encoded_replacement = gyp.common.EncodePOSIXShellList(replacement)
            else:
                encoded_replacement = replacement
            output = output[:replace_start] + str(encoded_replacement) + output[replace_end:]
        input_str = output
    if output == input:
        gyp.DebugOutput(gyp.DEBUG_VARIABLES, 'Found only identity matches on %r, avoiding infinite recursion.', output)
    else:
        gyp.DebugOutput(gyp.DEBUG_VARIABLES, 'Found output %r, recursing.', output)
        if type(output) is list:
            if output and type(output[0]) is list:
                pass
            else:
                new_output = []
                for item in output:
                    new_output.append(ExpandVariables(item, phase, variables, build_file))
                output = new_output
        else:
            output = ExpandVariables(output, phase, variables, build_file)
    if type(output) is list:
        for index, outstr in enumerate(output):
            if IsStrCanonicalInt(outstr):
                output[index] = int(outstr)
    elif IsStrCanonicalInt(output):
        output = int(output)
    return output