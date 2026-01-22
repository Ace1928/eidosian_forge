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
def _is_present(module, req, installed_pkgs, pkg_command):
    """Return whether or not package is installed."""
    for pkg in installed_pkgs:
        if '==' in pkg:
            pkg_name, pkg_version = pkg.split('==')
            pkg_name = Package.canonicalize_name(pkg_name)
        else:
            continue
        if pkg_name == req.package_name and req.is_satisfied_by(pkg_version):
            return True
    return False