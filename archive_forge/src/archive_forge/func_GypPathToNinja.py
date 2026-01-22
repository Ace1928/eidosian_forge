import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def GypPathToNinja(self, path, env=None):
    """Translate a gyp path to a ninja path, optionally expanding environment
        variable references in |path| with |env|.

        See the above discourse on path conversions."""
    if env:
        if self.flavor == 'mac':
            path = gyp.xcode_emulation.ExpandEnvVars(path, env)
        elif self.flavor == 'win':
            path = gyp.msvs_emulation.ExpandMacros(path, env)
    if path.startswith('$!'):
        expanded = self.ExpandSpecial(path)
        if self.flavor == 'win':
            expanded = os.path.normpath(expanded)
        return expanded
    if '$|' in path:
        path = self.ExpandSpecial(path)
    assert '$' not in path, path
    return os.path.normpath(os.path.join(self.build_to_base, path))