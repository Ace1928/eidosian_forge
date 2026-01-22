from __future__ import absolute_import, division, print_function
import argparse
import os
import re
import sys
import tempfile
import operator
import shlex
import traceback
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule, is_executable, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.six import PY3
def _get_cmd_options(module, cmd):
    thiscmd = cmd + ' --help'
    rc, stdout, stderr = module.run_command(thiscmd)
    if rc != 0:
        module.fail_json(msg='Could not get output from %s: %s' % (thiscmd, stdout + stderr))
    words = stdout.strip().split()
    cmd_options = [x for x in words if x.startswith('--')]
    return cmd_options