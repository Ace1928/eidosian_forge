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
def _SubninjaNameForArch(self, arch):
    output_file_base = os.path.splitext(self.output_file_name)[0]
    return f'{output_file_base}.{arch}.ninja'