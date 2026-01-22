import sys
from distutils.log import *  # noqa: F403
from distutils.log import Log as old_Log
from distutils.log import _global_log
from numpy.distutils.misc_util import (red_text, default_text, cyan_text,
def _fix_args(args, flag=1):
    if is_string(args):
        return args.replace('%', '%%')
    if flag and is_sequence(args):
        return tuple([_fix_args(a, flag=0) for a in args])
    return args