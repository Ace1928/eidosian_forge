import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
def DebugOutput(mode, message, *args):
    if 'all' in gyp.debug or mode in gyp.debug:
        ctx = ('unknown', 0, 'unknown')
        try:
            f = traceback.extract_stack(limit=2)
            if f:
                ctx = f[0][:3]
        except Exception:
            pass
        if args:
            message %= args
        print('%s:%s:%d:%s %s' % (mode.upper(), os.path.basename(ctx[0]), ctx[1], ctx[2], message))