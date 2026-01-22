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
def AppendPostbuildVariable(self, variables, spec, output, binary, is_command_start=False):
    """Adds a 'postbuild' variable if there is a postbuild for |output|."""
    postbuild = self.GetPostbuildCommand(spec, output, binary, is_command_start)
    if postbuild:
        variables.append(('postbuilds', postbuild))