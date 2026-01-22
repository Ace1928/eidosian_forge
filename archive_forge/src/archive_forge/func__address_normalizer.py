from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _address_normalizer(value):
    """
    Used to validate an address or network type and return it in a consistent format.
    This is being used for future use cases not currently available such as an address range.
    :param value: The string representation of an address or network.
    :return: The address or network in the normalized form.
    """
    try:
        vtype = ipaddr(value, 'type')
        if vtype == 'address' or vtype == 'network':
            v = ipaddr(value, 'subnet')
        else:
            return False
    except Exception:
        return False
    return v