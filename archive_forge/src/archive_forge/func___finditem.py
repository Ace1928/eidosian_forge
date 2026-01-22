from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def __finditem(obj, key):
    if key is None:
        return 'key_not_present'
    if key in obj:
        if obj[key] is None:
            return 'None'
        return obj[key]
    for dummy, val in obj.items():
        if isinstance(val, dict):
            item = __finditem(val, key)
            if item is not None:
                return item
    return None