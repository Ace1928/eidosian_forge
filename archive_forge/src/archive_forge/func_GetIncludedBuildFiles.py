import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def GetIncludedBuildFiles(build_file_path, aux_data, included=None):
    """Return a list of all build files included into build_file_path.

  The returned list will contain build_file_path as well as all other files
  that it included, either directly or indirectly.  Note that the list may
  contain files that were included into a conditional section that evaluated
  to false and was not merged into build_file_path's dict.

  aux_data is a dict containing a key for each build file or included build
  file.  Those keys provide access to dicts whose "included" keys contain
  lists of all other files included by the build file.

  included should be left at its default None value by external callers.  It
  is used for recursion.

  The returned list will not contain any duplicate entries.  Each build file
  in the list will be relative to the current directory.
  """
    if included is None:
        included = []
    if build_file_path in included:
        return included
    included.append(build_file_path)
    for included_build_file in aux_data[build_file_path].get('included', []):
        GetIncludedBuildFiles(included_build_file, aux_data, included)
    return included