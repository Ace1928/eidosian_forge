import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _AddObjectiveCGarbageCollectionFlags(self, flags):
    gc_policy = self._Settings().get('GCC_ENABLE_OBJC_GC', 'unsupported')
    if gc_policy == 'supported':
        flags.append('-fobjc-gc')
    elif gc_policy == 'required':
        flags.append('-fobjc-gc-only')