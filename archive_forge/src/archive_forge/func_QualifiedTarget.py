import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def QualifiedTarget(build_file, target, toolset):
    fully_qualified = build_file + ':' + target
    if toolset:
        fully_qualified = fully_qualified + '#' + toolset
    return fully_qualified