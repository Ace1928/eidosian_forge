from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from absl import flags
def _strip_undefok_args(undefok, args):
    """Returns a new list of args after removing flags in --undefok."""
    if undefok:
        undefok_names = set((name.strip() for name in undefok.split(',')))
        undefok_names |= set(('no' + name for name in undefok_names))
        args = [arg for arg in args if not _is_undefok(arg, undefok_names)]
    return args