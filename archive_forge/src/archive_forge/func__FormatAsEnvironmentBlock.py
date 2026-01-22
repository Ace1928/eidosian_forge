import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _FormatAsEnvironmentBlock(envvar_dict):
    """Format as an 'environment block' directly suitable for CreateProcess.
    Briefly this is a list of key=value\x00, terminated by an additional \x00. See
    CreateProcess documentation for more details."""
    block = ''
    nul = '\x00'
    for key, value in envvar_dict.items():
        block += key + '=' + value + nul
    block += nul
    return block