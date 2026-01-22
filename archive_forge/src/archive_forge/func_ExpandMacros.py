import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def ExpandMacros(string, expansions):
    """Expand $(Variable) per expansions dict. See MsvsSettings.GetVSMacroEnv
    for the canonical way to retrieve a suitable dict."""
    if '$' in string:
        for old, new in expansions.items():
            assert '$(' not in new, new
            string = string.replace(old, new)
    return string