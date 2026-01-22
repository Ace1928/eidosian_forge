import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetWrapperName(self):
    """Returns the directory name of the bundle represented by this target.
    Only valid for bundles."""
    assert self._IsBundle()
    return self.GetProductName() + self.GetWrapperExtension()