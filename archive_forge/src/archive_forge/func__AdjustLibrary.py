import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _AdjustLibrary(self, library, config_name=None):
    if library.endswith('.framework'):
        l_flag = '-framework ' + os.path.splitext(os.path.basename(library))[0]
    else:
        m = self.library_re.match(library)
        if m:
            l_flag = '-l' + m.group(1)
        else:
            l_flag = library
    sdk_root = self._SdkPath(config_name)
    if not sdk_root:
        sdk_root = ''
    library = l_flag.replace('$(SDKROOT)', sdk_root)
    if l_flag.startswith('$(SDKROOT)'):
        basename, ext = os.path.splitext(library)
        if ext == '.dylib' and (not os.path.exists(library)):
            tbd_library = basename + '.tbd'
            if os.path.exists(tbd_library):
                library = tbd_library
    return library