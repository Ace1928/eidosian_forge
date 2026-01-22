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
def add_module_options(parser):
    """Add options for commands that load modules"""
    module_path = C.config.get_configuration_definition('DEFAULT_MODULE_PATH').get('default', '')
    parser.add_argument('-M', '--module-path', dest='module_path', default=None, help='prepend colon-separated path(s) to module library (default=%s)' % module_path, type=unfrack_path(pathsep=True), action=PrependListAction)