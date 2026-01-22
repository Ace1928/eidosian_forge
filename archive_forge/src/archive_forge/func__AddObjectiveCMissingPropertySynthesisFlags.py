import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _AddObjectiveCMissingPropertySynthesisFlags(self, flags):
    if self._Test('CLANG_WARN_OBJC_MISSING_PROPERTY_SYNTHESIS', 'YES', default='NO'):
        flags.append('-Wobjc-missing-property-synthesis')