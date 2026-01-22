import abc
import contextlib
import hashlib
import os
from cinderclient.apiclient import base as common_base
from cinderclient import exceptions
from cinderclient import utils
def _format_sort_key_param(self, sort_key, resource_type=None):
    valid_sort_keys = SORT_KEY_VALUES
    if resource_type:
        add_sort_keys = SORT_KEY_ADD_VALUES.get(resource_type, None)
        if add_sort_keys:
            valid_sort_keys += add_sort_keys
    if sort_key in valid_sort_keys:
        return SORT_KEY_MAPPINGS.get(sort_key, sort_key)
    msg = 'sort_key must be one of the following: %s.' % ', '.join(valid_sort_keys)
    raise ValueError(msg)