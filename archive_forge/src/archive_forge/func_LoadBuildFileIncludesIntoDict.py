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
def LoadBuildFileIncludesIntoDict(subdict, subdict_path, data, aux_data, includes, check):
    includes_list = []
    if includes is not None:
        includes_list.extend(includes)
    if 'includes' in subdict:
        for include in subdict['includes']:
            relative_include = os.path.normpath(os.path.join(os.path.dirname(subdict_path), include))
            includes_list.append(relative_include)
        del subdict['includes']
    for include in includes_list:
        if 'included' not in aux_data[subdict_path]:
            aux_data[subdict_path]['included'] = []
        aux_data[subdict_path]['included'].append(include)
        gyp.DebugOutput(gyp.DEBUG_INCLUDES, "Loading Included File: '%s'", include)
        MergeDicts(subdict, LoadOneBuildFile(include, data, aux_data, None, False, check), subdict_path, include)
    for k, v in subdict.items():
        if type(v) is dict:
            LoadBuildFileIncludesIntoDict(v, subdict_path, data, aux_data, None, check)
        elif type(v) is list:
            LoadBuildFileIncludesIntoList(v, subdict_path, data, aux_data, check)