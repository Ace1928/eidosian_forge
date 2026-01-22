import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def EscapeShellArgument(s):
    """Quotes an argument so that it will be interpreted literally by a POSIX
    shell. Taken from
    http://stackoverflow.com/questions/35817/whats-the-best-way-to-escape-ossystem-calls-in-python
    """
    return "'" + s.replace("'", "'\\''") + "'"