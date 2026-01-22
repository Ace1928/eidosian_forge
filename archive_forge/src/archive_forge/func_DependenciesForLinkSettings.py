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
def DependenciesForLinkSettings(self, targets):
    """
    Returns a list of dependency targets whose link_settings should be merged
    into this target.
    """
    include_shared_libraries = targets[self.ref].get('allow_sharedlib_linksettings_propagation', True)
    return self._LinkDependenciesInternal(targets, include_shared_libraries)