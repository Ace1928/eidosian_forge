import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def AdjustLibraries(self, libraries, config_name=None):
    """Transforms entries like 'Cocoa.framework' in libraries into entries like
    '-framework Cocoa', 'libcrypto.dylib' into '-lcrypto', etc.
    """
    libraries = [self._AdjustLibrary(library, config_name) for library in libraries]
    return libraries