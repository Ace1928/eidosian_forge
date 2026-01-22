from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
class IllegalArgumentException(ValueError):
    """
    Exception raised when the function parameters:
        - not all passed
        - empty string
        - wrong type
    """
    pass