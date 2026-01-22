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
def add_inventory_options(parser):
    """Add options for commands that utilize inventory"""
    parser.add_argument('-i', '--inventory', '--inventory-file', dest='inventory', action='append', help='specify inventory host path or comma separated host list. --inventory-file is deprecated')
    parser.add_argument('--list-hosts', dest='listhosts', action='store_true', help='outputs a list of matching hosts; does not execute anything else')
    parser.add_argument('-l', '--limit', default=C.DEFAULT_SUBSET, dest='subset', help='further limit selected hosts to an additional pattern')