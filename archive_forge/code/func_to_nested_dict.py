from __future__ import absolute_import, division, print_function
import ssl
import atexit
import base64
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils._text import to_text, to_native
def to_nested_dict(vm_properties):
    """
    Parse properties from dot notation to dict

    """
    host_properties = {}
    for vm_prop_name, vm_prop_val in vm_properties.items():
        prop_parents = reversed(vm_prop_name.split('.'))
        prop_dict = parse_vim_property(vm_prop_val)
        for k in prop_parents:
            prop_dict = {k: prop_dict}
        host_properties = in_place_merge(host_properties, prop_dict)
    return host_properties