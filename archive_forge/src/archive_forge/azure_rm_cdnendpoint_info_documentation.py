from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import re

        Convert a Azure CDN endpoint object to dict.
        :param cdn: Azure CDN endpoint object
        :return: dict
        