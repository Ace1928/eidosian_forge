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
def ComputeOutputDir(params):
    """Returns the path from the toplevel_dir to the build output directory."""
    generator_dir = os.path.relpath(params['options'].generator_output or '.')
    output_dir = params.get('generator_flags', {}).get('output_dir', 'out')
    return os.path.normpath(os.path.join(generator_dir, output_dir))