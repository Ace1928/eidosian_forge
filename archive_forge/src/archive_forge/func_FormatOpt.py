import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
def FormatOpt(opt, value):
    if opt.startswith('--'):
        return f'{opt}={value}'
    return opt + value