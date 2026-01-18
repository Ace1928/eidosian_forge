from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def to_restriction_dict_list(self, restriction_obj_list):
    restrictions = []
    if restriction_obj_list:
        for r in restriction_obj_list:
            restriction = self.to_restriction_dict(r)
            if not self.is_azure_default_restriction(restriction):
                restrictions.append(restriction)
    return restrictions