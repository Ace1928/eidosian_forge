import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def Absolutify(self, path):
    """Convert a subdirectory-relative path into a base-relative path.
        Skips over paths that contain variables."""
    if '$(' in path:
        return path.rstrip('/')
    return os.path.normpath(os.path.join(self.path, path))