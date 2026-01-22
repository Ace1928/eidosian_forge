from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.common.text.converters import to_native
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from collections import namedtuple
import os
def _get_vm_pool(self):
    auth = self._get_connection_info()
    if not (auth.username and auth.password):
        raise AnsibleError('API Credentials missing. Check OpenNebula inventory file.')
    else:
        one_client = pyone.OneServer(auth.url, session=auth.username + ':' + auth.password)
    try:
        vm_pool = one_client.vmpool.infoextended(-2, -1, -1, 3)
    except Exception as e:
        raise AnsibleError('Something happened during XML-RPC call: {e}'.format(e=to_native(e)))
    return vm_pool