import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _IsIosWatchKitExtension(self):
    return int(self.spec.get('ios_watchkit_extension', 0)) != 0