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
def _exception2fail_json(msg='Generic failure: {0}'):
    """
    Decorator to convert Python exceptions into Ansible errors that can be reported to the user.
    """

    def decor(f):

        @wraps(f)
        def inner(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except Exception as e:
                err_msg = '{0}: {1}'.format(e.__class__.__name__, to_native(e))
                self.fail_from_exception(e, msg.format(err_msg))
        return inner
    return decor