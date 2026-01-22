import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _SdkPath(self, configname=None):
    sdk_root = self._SdkRoot(configname)
    if sdk_root.startswith('/'):
        return sdk_root
    return self._XcodeSdkPath(sdk_root)