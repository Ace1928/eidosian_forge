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
def ProcessToolsetsInDict(data):
    if 'targets' in data:
        target_list = data['targets']
        new_target_list = []
        for target in target_list:
            if 'toolset' in target and 'toolsets' not in target:
                new_target_list.append(target)
                continue
            if multiple_toolsets:
                toolsets = target.get('toolsets', ['target'])
            else:
                toolsets = ['target']
            if 'toolsets' in target:
                del target['toolsets']
            if len(toolsets) > 0:
                for build in toolsets[1:]:
                    new_target = gyp.simple_copy.deepcopy(target)
                    new_target['toolset'] = build
                    new_target_list.append(new_target)
                target['toolset'] = toolsets[0]
                new_target_list.append(target)
        data['targets'] = new_target_list
    if 'conditions' in data:
        for condition in data['conditions']:
            if type(condition) is list:
                for condition_dict in condition[1:]:
                    if type(condition_dict) is dict:
                        ProcessToolsetsInDict(condition_dict)