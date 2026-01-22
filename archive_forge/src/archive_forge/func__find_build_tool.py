import os
import re
import sys
def _find_build_tool(toolname):
    """Find a build tool on current path or using xcrun"""
    return _find_executable(toolname) or _read_output('/usr/bin/xcrun -find %s' % (toolname,)) or ''