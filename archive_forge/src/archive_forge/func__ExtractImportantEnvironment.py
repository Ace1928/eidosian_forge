import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _ExtractImportantEnvironment(output_of_set):
    """Extracts environment variables required for the toolchain to run from
    a textual dump output by the cmd.exe 'set' command."""
    envvars_to_save = ('goma_.*', 'include', 'lib', 'libpath', 'path', 'pathext', 'systemroot', 'temp', 'tmp')
    env = {}
    if output_of_set.count('=') == 0:
        raise Exception('Invalid output_of_set. Value is:\n%s' % output_of_set)
    for line in output_of_set.splitlines():
        for envvar in envvars_to_save:
            if re.match(envvar + '=', line.lower()):
                var, setting = line.split('=', 1)
                if envvar == 'path':
                    setting = os.path.dirname(sys.executable) + os.pathsep + setting
                env[var.upper()] = setting
                break
    for required in ('SYSTEMROOT', 'TEMP', 'TMP'):
        if required not in env:
            raise Exception('Environment variable "%s" required to be set to valid path' % required)
    return env