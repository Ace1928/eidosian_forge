import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _SdkRoot(self, configname):
    if configname is None:
        configname = self.configname
    return self.GetPerConfigSetting('SDKROOT', configname, default='')