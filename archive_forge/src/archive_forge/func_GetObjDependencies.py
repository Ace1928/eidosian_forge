import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetObjDependencies(self, sources, objs, arch=None):
    """Given a list of source files and the corresponding object files, returns
    a list of (source, object, gch) tuples, where |gch| is the build-directory
    relative path to the gch file each object file depends on.  |compilable[i]|
    has to be the source file belonging to |objs[i]|."""
    if not self.header or not self.compile_headers:
        return []
    result = []
    for source, obj in zip(sources, objs):
        ext = os.path.splitext(source)[1]
        lang = {'.c': 'c', '.cpp': 'cc', '.cc': 'cc', '.cxx': 'cc', '.m': 'm', '.mm': 'mm'}.get(ext, None)
        if lang:
            result.append((source, obj, self._Gch(lang, arch)))
    return result