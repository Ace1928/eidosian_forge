from __future__ import absolute_import, division, print_function
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
def validate_spec():
    return dict(fail_on_error=dict(type='bool'), version=dict(), strict=dict(type='bool', default=True))