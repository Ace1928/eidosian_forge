import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetCflagsObjC(self, configname):
    """Returns flags that need to be added to .m compilations."""
    self.configname = configname
    cflags_objc = []
    self._AddObjectiveCGarbageCollectionFlags(cflags_objc)
    self._AddObjectiveCARCFlags(cflags_objc)
    self._AddObjectiveCMissingPropertySynthesisFlags(cflags_objc)
    self.configname = None
    return cflags_objc