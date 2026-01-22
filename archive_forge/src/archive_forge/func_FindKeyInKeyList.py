from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
def FindKeyInKeyList(key_arg, profile_keys):
    """Return the fingerprint of an SSH key that matches the key argument."""
    key = profile_keys.get(key_arg)
    if key:
        return key_arg
    key_split = key_arg.split()
    if not key_split:
        return None
    if len(key_split) == 1:
        key_value = key_split[0]
    else:
        key_value = key_split[1]
    for fingerprint, ssh_key in profile_keys.items():
        if key_value in ssh_key:
            return fingerprint