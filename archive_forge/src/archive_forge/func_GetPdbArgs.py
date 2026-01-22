from __future__ import print_function
import os
import re
import sys
def GetPdbArgs(python):
    """Try to get the path to pdb.py and return it in a list.

  Args:
    python: The full path to a Python executable.

  Returns:
    A list of strings. If a relevant pdb.py was found, this will be
    ['/path/to/pdb.py']; if not, return ['-m', 'pdb'] and hope for the best.
    (This latter technique will fail for Python 2.2.)
  """
    components = python.split('/')
    if len(components) >= 2:
        pdb_path = '/'.join(components[0:-2] + ['lib'] + components[-1:] + ['pdb.py'])
        if os.access(pdb_path, os.R_OK):
            return [pdb_path]
    return ['-m', 'pdb']