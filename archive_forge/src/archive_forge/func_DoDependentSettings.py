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
def DoDependentSettings(key, flat_list, targets, dependency_nodes):
    for target in flat_list:
        target_dict = targets[target]
        build_file = gyp.common.BuildFile(target)
        if key == 'all_dependent_settings':
            dependencies = dependency_nodes[target].DeepDependencies()
        elif key == 'direct_dependent_settings':
            dependencies = dependency_nodes[target].DirectAndImportedDependencies(targets)
        elif key == 'link_settings':
            dependencies = dependency_nodes[target].DependenciesForLinkSettings(targets)
        else:
            raise GypError("DoDependentSettings doesn't know how to determine dependencies for " + key)
        for dependency in dependencies:
            dependency_dict = targets[dependency]
            if key not in dependency_dict:
                continue
            dependency_build_file = gyp.common.BuildFile(dependency)
            MergeDicts(target_dict, dependency_dict[key], build_file, dependency_build_file)