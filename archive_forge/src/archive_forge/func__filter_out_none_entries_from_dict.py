from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def _filter_out_none_entries_from_dict(self, adict, allow_empty_list_or_dict):
    """take a dict as input and return a dict without keys whose values are None
           return empty dicts or lists if allow_empty_list_or_dict otherwise skip empty dicts or lists.
        """
    result = {}
    for key, value in adict.items():
        if isinstance(value, (list, dict)):
            sub = self.filter_out_none_entries(value, allow_empty_list_or_dict)
            if sub or allow_empty_list_or_dict:
                result[key] = sub
        elif value is not None:
            result[key] = value
    return result