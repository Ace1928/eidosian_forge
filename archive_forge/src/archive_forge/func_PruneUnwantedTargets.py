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
def PruneUnwantedTargets(targets, flat_list, dependency_nodes, root_targets, data):
    """Return only the targets that are deep dependencies of |root_targets|."""
    qualified_root_targets = []
    for target in root_targets:
        target = target.strip()
        qualified_targets = gyp.common.FindQualifiedTargets(target, flat_list)
        if not qualified_targets:
            raise GypError('Could not find target %s' % target)
        qualified_root_targets.extend(qualified_targets)
    wanted_targets = {}
    for target in qualified_root_targets:
        wanted_targets[target] = targets[target]
        for dependency in dependency_nodes[target].DeepDependencies():
            wanted_targets[dependency] = targets[dependency]
    wanted_flat_list = [t for t in flat_list if t in wanted_targets]
    for build_file in data['target_build_files']:
        if 'targets' not in data[build_file]:
            continue
        new_targets = []
        for target in data[build_file]['targets']:
            qualified_name = gyp.common.QualifiedTarget(build_file, target['target_name'], target['toolset'])
            if qualified_name in wanted_targets:
                new_targets.append(target)
        data[build_file]['targets'] = new_targets
    return (wanted_targets, wanted_flat_list)