import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def _supportsSymlinks(self) -> bool:
    """
        Check for symlink support usable for Twisted's purposes.

        @return: C{True} if symlinks are supported on the current platform,
                 otherwise C{False}.
        """
    if self.isWindows():
        return False
    else:
        try:
            os.symlink
        except AttributeError:
            return False
        else:
            return True