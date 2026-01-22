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
def SetGeneratorGlobals(generator_input_info):
    global path_sections
    path_sections = set(base_path_sections)
    path_sections.update(generator_input_info['path_sections'])
    global non_configuration_keys
    non_configuration_keys = base_non_configuration_keys[:]
    non_configuration_keys.extend(generator_input_info['non_configuration_keys'])
    global multiple_toolsets
    multiple_toolsets = generator_input_info['generator_supports_multiple_toolsets']
    global generator_filelist_paths
    generator_filelist_paths = generator_input_info['generator_filelist_paths']