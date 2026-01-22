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
def _have_pip_module():
    """Return True if the `pip` module can be found using the current Python interpreter, otherwise return False."""
    try:
        from importlib.util import find_spec
    except ImportError:
        find_spec = None
    if find_spec:
        try:
            found = bool(find_spec('pip'))
        except Exception:
            found = False
    else:
        import imp
        try:
            imp.find_module('pip')
        except Exception:
            found = False
        else:
            found = True
    return found