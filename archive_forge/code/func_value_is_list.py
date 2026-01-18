from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six.moves.collections_abc import MutableMapping
def value_is_list(camel_list):
    checked_list = []
    for item in camel_list:
        if isinstance(item, dict):
            checked_list.append(camel_dict_to_snake_dict(item, reversible))
        elif isinstance(item, list):
            checked_list.append(value_is_list(item))
        else:
            checked_list.append(item)
    return checked_list