from __future__ import absolute_import, division, print_function
from functools import wraps
from ansible import errors
from ansible.errors import AnsibleError
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import ensure_text
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
def _need_netaddr(f_name, *args, **kwargs):
    raise errors.AnsibleFilterError(missing_required_lib('netaddr'))