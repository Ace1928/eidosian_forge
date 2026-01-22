from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def _check_dsa_consistency(key_public_data, key_private_data):
    p = key_public_data.get('p')
    q = key_public_data.get('q')
    g = key_public_data.get('g')
    y = key_public_data.get('y')
    x = key_private_data.get('x')
    for v in (p, q, g, y, x):
        if v is None:
            return None
    if g < 2 or g >= p - 1:
        return False
    if x < 1 or x >= q:
        return False
    if (p - 1) % q != 0:
        return False
    if binary_exp_mod(g, q, p) != 1:
        return False
    if binary_exp_mod(g, x, p) != y:
        return False
    if quick_is_not_prime(q) or quick_is_not_prime(p):
        return False
    return True