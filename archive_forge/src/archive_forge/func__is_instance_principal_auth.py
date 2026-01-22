from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def _is_instance_principal_auth(module):
    instance_principal_auth = 'auth_type' in module.params and module.params['auth_type'] == 'instance_principal'
    if not instance_principal_auth:
        instance_principal_auth = 'OCI_ANSIBLE_AUTH_TYPE' in os.environ and os.environ['OCI_ANSIBLE_AUTH_TYPE'] == 'instance_principal'
    return instance_principal_auth