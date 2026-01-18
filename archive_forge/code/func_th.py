from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
def th(number):
    abs_number = abs(number)
    mod_10 = abs_number % 10
    mod_100 = abs_number % 100
    if mod_100 not in (11, 12, 13):
        if mod_10 == 1:
            return 'st'
        if mod_10 == 2:
            return 'nd'
        if mod_10 == 3:
            return 'rd'
    return 'th'