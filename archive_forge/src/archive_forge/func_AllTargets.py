import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def AllTargets(target_list, target_dicts, build_file):
    """Returns all targets (direct and dependencies) for the specified build_file.
  """
    bftargets = BuildFileTargets(target_list, build_file)
    deptargets = DeepDependencyTargets(target_dicts, bftargets)
    return bftargets + deptargets