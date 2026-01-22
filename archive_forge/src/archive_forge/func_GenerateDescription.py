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
def GenerateDescription(self, verb, message, fallback):
    """Generate and return a description of a build step.

        |verb| is the short summary, e.g. ACTION or RULE.
        |message| is a hand-written description, or None if not available.
        |fallback| is the gyp-level name of the step, usable as a fallback.
        """
    if self.toolset != 'target':
        verb += '(%s)' % self.toolset
    if message:
        return f'{verb} {self.ExpandSpecial(message)}'
    else:
        return f'{verb} {self.name}: {fallback}'