import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def _InstallableTargetInstallPath(self):
    """Returns the location of the final output for an installable target."""
    return '$(builddir)/' + self.alias