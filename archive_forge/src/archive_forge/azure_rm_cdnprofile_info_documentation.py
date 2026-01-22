from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import re

        Convert a CDN profile object to dict.
        :param cdn: CDN profile object
        :return: dict
        