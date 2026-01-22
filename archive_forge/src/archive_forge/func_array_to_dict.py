from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def array_to_dict(array):
    """Converts list object to dictionary object, including any nested properties on elements."""
    new = {}
    for index, item in enumerate(array):
        new[index] = deepcopy(item)
        if isinstance(item, dict):
            for nested in item:
                if isinstance(item[nested], list):
                    new[index][nested] = array_to_dict(item[nested])
    return new