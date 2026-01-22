import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _Gch(self, lang, arch):
    """Returns the actual file name of the prefix header for language |lang|."""
    assert self.compile_headers
    return self._CompiledHeader(lang, arch) + '.gch'