from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.waiter import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.resource import (
def _continue_or_fail(error):
    if multiple_scale and continue_on_error:
        if 'errors' not in return_attributes:
            return_attributes['errors'] = []
        return_attributes['errors'].append({'error': error, 'failed': True})
    else:
        module.fail_json(msg=error, **return_attributes)