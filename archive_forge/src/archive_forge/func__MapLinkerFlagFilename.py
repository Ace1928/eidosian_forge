import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _MapLinkerFlagFilename(self, ldflag, gyp_to_build_path):
    """Checks if ldflag contains a filename and if so remaps it from
    gyp-directory-relative to build-directory-relative."""
    LINKER_FILE = '(\\S+)'
    WORD = '\\S+'
    linker_flags = [['-exported_symbols_list', LINKER_FILE], ['-unexported_symbols_list', LINKER_FILE], ['-reexported_symbols_list', LINKER_FILE], ['-sectcreate', WORD, WORD, LINKER_FILE]]
    for flag_pattern in linker_flags:
        regex = re.compile('(?:-Wl,)?' + '[ ,]'.join(flag_pattern))
        m = regex.match(ldflag)
        if m:
            ldflag = ldflag[:m.start(1)] + gyp_to_build_path(m.group(1)) + ldflag[m.end(1):]
    if ldflag.startswith('-L'):
        ldflag = '-L' + gyp_to_build_path(ldflag[len('-L'):])
    return ldflag