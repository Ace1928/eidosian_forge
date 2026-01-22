from __future__ import (absolute_import, division, print_function)
import copy
import operator
import argparse
import os
import os.path
import sys
import time
from jinja2 import __version__ as j2_version
import ansible
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.yaml import HAS_LIBYAML, yaml_load
from ansible.release import __version__
from ansible.utils.path import unfrackpath
def add_output_options(parser):
    """Add options for commands which can change their output"""
    parser.add_argument('-o', '--one-line', dest='one_line', action='store_true', help='condense output')
    parser.add_argument('-t', '--tree', dest='tree', default=None, help='log output to this directory')