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
def add_check_options(parser):
    """Add options for commands which can run with diagnostic information of tasks"""
    parser.add_argument('-C', '--check', default=False, dest='check', action='store_true', help="don't make any changes; instead, try to predict some of the changes that may occur")
    parser.add_argument('-D', '--diff', default=C.DIFF_ALWAYS, dest='diff', action='store_true', help='when changing (small) files and templates, show the differences in those files; works great with --check')