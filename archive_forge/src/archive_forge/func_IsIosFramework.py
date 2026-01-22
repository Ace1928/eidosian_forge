import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def IsIosFramework(self):
    return self.spec['type'] == 'shared_library' and self._IsBundle() and self.isIOS