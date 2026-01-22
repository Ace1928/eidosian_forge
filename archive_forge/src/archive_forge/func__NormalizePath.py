from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
def _NormalizePath(basedir, pathname):
    """Get the absolute path from a unix-style relative path.

  Args:
    basedir: (str) Platform-specific encoding of the base directory.
    pathname: (str) A unix-style (forward slash separated) path relative to
      the runtime definition root directory.

  Returns:
    (str) An absolute path conforming to the conventions of the operating
    system.  Note: in order for this to work, 'pathname' must not contain
    any characters with special meaning in any of the targeted operating
    systems.  Keep those names simple.
  """
    components = pathname.split('/')
    return os.path.join(basedir, *components)