import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def EscapeMakeVariableExpansion(s):
    """Make has its own variable expansion syntax using $. We must escape it for
    string to be interpreted literally."""
    return s.replace('$', '$$')