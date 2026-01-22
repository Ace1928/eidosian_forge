from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _range_checker(ip_check, first, last):
    """
    Tests whether an ip address is within the bounds of the first and last address.
    :param ip_check: The ip to test if it is within first and last.
    :param first: The first IP in the range to test against.
    :param last: The last IP in the range to test against.
    :return: bool
    """
    if first <= ip_check <= last:
        return True
    else:
        return False