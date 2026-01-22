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
def LoadTargetBuildFile(build_file_path, data, aux_data, variables, includes, depth, check, load_dependencies):
    if depth:
        d = gyp.common.RelativePath(depth, os.path.dirname(build_file_path))
        if d == '':
            variables['DEPTH'] = '.'
        else:
            variables['DEPTH'] = d.replace('\\', '/')
    if 'target_build_files' in data:
        if build_file_path in data['target_build_files']:
            return False
        data['target_build_files'].add(build_file_path)
    gyp.DebugOutput(gyp.DEBUG_INCLUDES, "Loading Target Build File '%s'", build_file_path)
    build_file_data = LoadOneBuildFile(build_file_path, data, aux_data, includes, True, check)
    build_file_data['_DEPTH'] = depth
    if 'included_files' in build_file_data:
        raise GypError(build_file_path + ' must not contain included_files key')
    included = GetIncludedBuildFiles(build_file_path, aux_data)
    build_file_data['included_files'] = []
    for included_file in included:
        included_relative = gyp.common.RelativePath(included_file, os.path.dirname(build_file_path))
        build_file_data['included_files'].append(included_relative)
    ProcessToolsetsInDict(build_file_data)
    ProcessVariablesAndConditionsInDict(build_file_data, PHASE_EARLY, variables, build_file_path)
    ProcessToolsetsInDict(build_file_data)
    if 'target_defaults' in build_file_data:
        if 'targets' not in build_file_data:
            raise GypError('Unable to find targets in build file %s' % build_file_path)
        index = 0
        while index < len(build_file_data['targets']):
            old_target_dict = build_file_data['targets'][index]
            new_target_dict = gyp.simple_copy.deepcopy(build_file_data['target_defaults'])
            MergeDicts(new_target_dict, old_target_dict, build_file_path, build_file_path)
            build_file_data['targets'][index] = new_target_dict
            index += 1
        del build_file_data['target_defaults']
    dependencies = []
    if 'targets' in build_file_data:
        for target_dict in build_file_data['targets']:
            if 'dependencies' not in target_dict:
                continue
            for dependency in target_dict['dependencies']:
                dependencies.append(gyp.common.ResolveTarget(build_file_path, dependency, None)[0])
    if load_dependencies:
        for dependency in dependencies:
            try:
                LoadTargetBuildFile(dependency, data, aux_data, variables, includes, depth, check, load_dependencies)
            except Exception as e:
                gyp.common.ExceptionAppend(e, 'while loading dependencies of %s' % build_file_path)
                raise
    else:
        return (build_file_path, dependencies)