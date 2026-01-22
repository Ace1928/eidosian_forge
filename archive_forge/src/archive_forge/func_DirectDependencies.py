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
def DirectDependencies(self, dependencies=None):
    """Returns a list of just direct dependencies."""
    if dependencies is None:
        dependencies = []
    for dependency in self.dependencies:
        if dependency.ref and dependency.ref not in dependencies:
            dependencies.append(dependency.ref)
    return dependencies