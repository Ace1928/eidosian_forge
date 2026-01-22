import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def AddImplicitPostbuilds(self, configname, output, output_binary, postbuilds=[], quiet=False):
    """Returns a list of shell commands that should run before and after
    |postbuilds|."""
    assert output_binary is not None
    pre = self._GetTargetPostbuilds(configname, output, output_binary, quiet)
    post = self._GetIOSPostbuilds(configname, output_binary)
    return pre + postbuilds + post