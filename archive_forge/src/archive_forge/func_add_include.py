import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def add_include(self, args, unknown):
    """Add include directories.

        :param args: args as returned by get_r_flags().
        :param unknown: unknown arguments a returned by get_r_flags()."""
    if args.I is None:
        warnings.warn('No include specified')
    else:
        self.include_dirs.extend(args.I)
    self.extra_compile_args.extend(unknown)