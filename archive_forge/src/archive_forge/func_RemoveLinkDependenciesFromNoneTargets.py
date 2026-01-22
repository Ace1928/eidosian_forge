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
def RemoveLinkDependenciesFromNoneTargets(targets):
    """Remove dependencies having the 'link_dependency' attribute from the 'none'
  targets."""
    for target_name, target_dict in targets.items():
        for dependency_key in dependency_sections:
            dependencies = target_dict.get(dependency_key, [])
            if dependencies:
                for t in dependencies:
                    if target_dict.get('type', None) == 'none':
                        if targets[t].get('variables', {}).get('link_dependency', 0):
                            target_dict[dependency_key] = Filter(target_dict[dependency_key], t)