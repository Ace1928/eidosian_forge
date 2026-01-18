from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
def to_dict_list(self, raws):
    return [nic_to_dict(item) for item in raws] if raws else []