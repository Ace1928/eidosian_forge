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
def WriteCollapsedDependencies(self, name, targets, order_only=None):
    """Given a list of targets, return a path for a single file
        representing the result of building all the targets or None.

        Uses a stamp file if necessary."""
    assert targets == [item for item in targets if item], targets
    if len(targets) == 0:
        assert not order_only
        return None
    if len(targets) > 1 or order_only:
        stamp = self.GypPathToUniqueOutput(name + '.stamp')
        targets = self.ninja.build(stamp, 'stamp', targets, order_only=order_only)
        self.ninja.newline()
    return targets[0]