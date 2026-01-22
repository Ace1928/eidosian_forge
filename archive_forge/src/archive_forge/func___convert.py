from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __convert(self, val):
    """Convert unserializable data."""
    try:
        if isinstance(val, Decimal):
            val = float(val)
        else:
            val = int(val)
    except ValueError:
        pass
    except TypeError:
        pass
    return val