from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
def _api_permission_denied_handler(name):
    """Return decorator which catches #403 errors"""

    def inner(func):

        def wrapper(module, fusion, *args, **kwargs):
            try:
                return func(module, fusion, *args, **kwargs)
            except purefusion.rest.ApiException as exc:
                if exc.status == http.HTTPStatus.FORBIDDEN:
                    module.warn(f'Cannot get [{name} dict], reason: Permission denied')
                    return None
                else:
                    raise exc
        return wrapper
    return inner