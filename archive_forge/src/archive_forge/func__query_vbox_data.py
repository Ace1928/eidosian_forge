from __future__ import (absolute_import, division, print_function)
import os
from subprocess import Popen, PIPE
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.common.process import get_bin_path
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _query_vbox_data(self, host, property_path):
    ret = None
    try:
        cmd = [self._vbox_path, b'guestproperty', b'get', to_bytes(host, errors='surrogate_or_strict'), to_bytes(property_path, errors='surrogate_or_strict')]
        x = Popen(cmd, stdout=PIPE)
        ipinfo = to_text(x.stdout.read(), errors='surrogate_or_strict')
        if 'Value' in ipinfo:
            a, ip = ipinfo.split(':', 1)
            ret = ip.strip()
    except Exception:
        pass
    return ret