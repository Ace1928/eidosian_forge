import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetMachOType(self):
    """Returns the MACH_O_TYPE of this target."""
    if not self._IsBundle() and self.spec['type'] == 'executable':
        return ''
    return {'executable': 'mh_execute', 'static_library': 'staticlib', 'shared_library': 'mh_dylib', 'loadable_module': 'mh_bundle'}[self.spec['type']]