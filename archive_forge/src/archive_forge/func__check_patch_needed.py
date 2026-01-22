from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
def _check_patch_needed(introduced_version=None, fixed_version=None, plugins=None):
    """
    Decorator to check whether a specific apidoc patch is required.

    :param introduced_version: The version of Foreman the API bug was introduced.
    :type introduced_version: str, optional
    :param fixed_version: The version of Foreman the API bug was fixed.
    :type fixed_version: str, optional
    :param plugins: Which plugins are required for this patch.
    :type plugins: list, optional
    """

    def decor(f):

        @wraps(f)
        def inner(self, *args, **kwargs):
            if plugins is not None and (not all((self.has_plugin(plugin) for plugin in plugins))):
                return
            if fixed_version is not None and self.foreman_version >= LooseVersion(fixed_version):
                return
            if introduced_version is not None and self.foreman_version < LooseVersion(introduced_version):
                return
            return f(self, *args, **kwargs)
        return inner
    return decor