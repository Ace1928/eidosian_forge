from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
import re

        Convert a Traffic Manager profile object to dict.
        :param tm: Traffic Manager profile object
        :return: dict
        