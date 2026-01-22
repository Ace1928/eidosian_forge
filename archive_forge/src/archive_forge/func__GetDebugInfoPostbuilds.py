import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetDebugInfoPostbuilds(self, configname, output, output_binary, quiet):
    """Returns a list of shell commands that contain the shell commands
    necessary to massage this target's debug information. These should be run
    as postbuilds before the actual postbuilds run."""
    self.configname = configname
    result = []
    if self._Test('GCC_GENERATE_DEBUGGING_SYMBOLS', 'YES', default='YES') and self._Test('DEBUG_INFORMATION_FORMAT', 'dwarf-with-dsym', default='dwarf') and (self.spec['type'] != 'static_library'):
        if not quiet:
            result.append('echo DSYMUTIL\\(%s\\)' % self.spec['target_name'])
        result.append('dsymutil {} -o {}'.format(output_binary, output + '.dSYM'))
    self.configname = None
    return result