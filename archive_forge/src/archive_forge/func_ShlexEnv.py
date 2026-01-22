import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
def ShlexEnv(env_name):
    flags = os.environ.get(env_name, [])
    if flags:
        flags = shlex.split(flags)
    return flags