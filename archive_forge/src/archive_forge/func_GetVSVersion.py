import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetVSVersion(generator_flags):
    global vs_version
    if not vs_version:
        vs_version = gyp.MSVSVersion.SelectVisualStudioVersion(generator_flags.get('msvs_version', 'auto'), allow_fallback=False)
    return vs_version