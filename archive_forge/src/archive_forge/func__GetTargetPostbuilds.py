import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetTargetPostbuilds(self, configname, output, output_binary, quiet=False):
    """Returns a list of shell commands that contain the shell commands
    to run as postbuilds for this target, before the actual postbuilds."""
    return self._GetDebugInfoPostbuilds(configname, output, output_binary, quiet) + self._GetStripPostbuilds(configname, output_binary, quiet)