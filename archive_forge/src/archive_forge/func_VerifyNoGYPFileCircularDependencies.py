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
def VerifyNoGYPFileCircularDependencies(targets):
    dependency_nodes = {}
    for target in targets:
        build_file = gyp.common.BuildFile(target)
        if build_file not in dependency_nodes:
            dependency_nodes[build_file] = DependencyGraphNode(build_file)
    for target, spec in targets.items():
        build_file = gyp.common.BuildFile(target)
        build_file_node = dependency_nodes[build_file]
        target_dependencies = spec.get('dependencies', [])
        for dependency in target_dependencies:
            try:
                dependency_build_file = gyp.common.BuildFile(dependency)
            except GypError as e:
                gyp.common.ExceptionAppend(e, 'while computing dependencies of .gyp file %s' % build_file)
                raise
            if dependency_build_file == build_file:
                continue
            dependency_node = dependency_nodes.get(dependency_build_file)
            if not dependency_node:
                raise GypError("Dependency '%s' not found" % dependency_build_file)
            if dependency_node not in build_file_node.dependencies:
                build_file_node.dependencies.append(dependency_node)
                dependency_node.dependents.append(build_file_node)
    root_node = DependencyGraphNode(None)
    for build_file_node in dependency_nodes.values():
        if len(build_file_node.dependencies) == 0:
            build_file_node.dependencies.append(root_node)
            root_node.dependents.append(build_file_node)
    flat_list = root_node.FlattenToList()
    if len(flat_list) != len(dependency_nodes):
        if not root_node.dependents:
            file_node = next(iter(dependency_nodes.values()))
            file_node.dependencies.append(root_node)
            root_node.dependents.append(file_node)
        cycles = []
        for cycle in root_node.FindCycles():
            paths = [node.ref for node in cycle]
            cycles.append('Cycle: %s' % ' -> '.join(paths))
        raise DependencyGraphNode.CircularException('Cycles in .gyp file dependency graph detected:\n' + '\n'.join(cycles))