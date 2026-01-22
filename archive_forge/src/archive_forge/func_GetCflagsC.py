import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetCflagsC(self, configname):
    """Returns flags that need to be added to .c, and .m compilations."""
    self.configname = configname
    cflags_c = []
    if self._Settings().get('GCC_C_LANGUAGE_STANDARD', '') == 'ansi':
        cflags_c.append('-ansi')
    else:
        self._Appendf(cflags_c, 'GCC_C_LANGUAGE_STANDARD', '-std=%s')
    cflags_c += self._Settings().get('OTHER_CFLAGS', [])
    self.configname = None
    return cflags_c