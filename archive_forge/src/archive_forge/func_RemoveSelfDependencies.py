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
def RemoveSelfDependencies(targets):
    """Remove self dependencies from targets that have the prune_self_dependency
  variable set."""
    for target_name, target_dict in targets.items():
        for dependency_key in dependency_sections:
            dependencies = target_dict.get(dependency_key, [])
            if dependencies:
                for t in dependencies:
                    if t == target_name:
                        if targets[t].get('variables', {}).get('prune_self_dependency', 0):
                            target_dict[dependency_key] = Filter(dependencies, target_name)