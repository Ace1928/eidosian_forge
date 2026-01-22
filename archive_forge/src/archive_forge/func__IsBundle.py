import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _IsBundle(self):
    return int(self.spec.get('mac_bundle', 0)) != 0 or self._IsXCTest() or self._IsXCUiTest()