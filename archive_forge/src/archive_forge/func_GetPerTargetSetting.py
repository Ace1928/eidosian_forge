import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetPerTargetSetting(self, setting, default=None):
    """Tries to get xcode_settings.setting from spec. Assumes that the setting
       has the same value in all configurations and throws otherwise."""
    is_first_pass = True
    result = None
    for configname in sorted(self.xcode_settings.keys()):
        if is_first_pass:
            result = self.xcode_settings[configname].get(setting, None)
            is_first_pass = False
        else:
            assert result == self.xcode_settings[configname].get(setting, None), "Expected per-target setting for '%s', got per-config setting (target %s)" % (setting, self.spec['target_name'])
    if result is None:
        return default
    return result