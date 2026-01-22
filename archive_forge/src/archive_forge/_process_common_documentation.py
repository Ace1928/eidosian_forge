import subprocess
import shlex
import sys
import os
from IPython.utils import py3compat
Split a command line's arguments in a shell-like manner.

    This is a modified version of the standard library's shlex.split()
    function, but with a default of posix=False for splitting, so that quotes
    in inputs are respected.

    if strict=False, then any errors shlex.split would raise will result in the
    unparsed remainder being the last element of the list, rather than raising.
    This is because we sometimes use arg_split to parse things other than
    command-line args.
    