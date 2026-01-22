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
def add_runas_prompt_options(parser, runas_group=None):
    """
    Add options for commands which need to prompt for privilege escalation credentials

    Note that add_runas_options() includes these options already.  Only one of the two functions
    should be used.
    """
    if runas_group is not None:
        parser.add_argument_group(runas_group)
    runas_pass_group = parser.add_mutually_exclusive_group()
    runas_pass_group.add_argument('-K', '--ask-become-pass', dest='become_ask_pass', action='store_true', default=C.DEFAULT_BECOME_ASK_PASS, help='ask for privilege escalation password')
    runas_pass_group.add_argument('--become-password-file', '--become-pass-file', default=C.BECOME_PASSWORD_FILE, dest='become_password_file', help='Become password file', type=unfrack_path(), action='store')
    parser.add_argument_group(runas_pass_group)