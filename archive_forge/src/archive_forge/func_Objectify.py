import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def Objectify(self, path):
    """Convert a path to its output directory form."""
    if '$(' in path:
        path = path.replace('$(obj)/', '$(obj).%s/$(TARGET)/' % self.toolset)
    if '$(obj)' not in path:
        path = f'$(obj).{self.toolset}/$(TARGET)/{path}'
    return path