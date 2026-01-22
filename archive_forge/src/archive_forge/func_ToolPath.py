import errno
import os
import re
import subprocess
import sys
import glob
def ToolPath(self, tool):
    """Returns the path to a given compiler tool. """
    return os.path.normpath(os.path.join(self.path, 'VC/bin', tool))