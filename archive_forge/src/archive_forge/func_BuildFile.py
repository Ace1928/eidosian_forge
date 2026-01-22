import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def BuildFile(fully_qualified_target):
    return ParseQualifiedTarget(fully_qualified_target)[0]