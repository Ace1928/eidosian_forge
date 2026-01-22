from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient

        Get the resource scope of the lock management.
        '/subscriptions/{subscriptionId}' for subscriptions,
        '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}' for resource groups,
        '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/{namespace}/{resourceType}/{resourceName}' for resources.
        