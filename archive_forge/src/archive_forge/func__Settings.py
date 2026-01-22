import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _Settings(self):
    assert self.configname
    return self.xcode_settings[self.configname]