import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def add_lib(self, args, unknown, ignore=('R',)):
    """Add libraries.

        :param args: args as returned by get_r_flags().
        :param unknown: unknown arguments a returned by get_r_flags()."""
    if args.L is None:
        if args.l is None:
            warnings.warn('No libraries as -l arguments to the compiler.')
        else:
            self.libraries.extend([x for x in args.l if x not in ignore])
    else:
        self.library_dirs.extend(args.L)
        self.libraries.extend(args.l)
    self.extra_link_args.extend(unknown)