import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _ExtractCLPath(output_of_where):
    """Gets the path to cl.exe based on the output of calling the environment
    setup batch file, followed by the equivalent of `where`."""
    for line in output_of_where.strip().splitlines():
        if line.startswith('LOC:'):
            return line[len('LOC:'):].strip()