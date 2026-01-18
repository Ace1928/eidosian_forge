from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def user_to_dict(self, object):
    return dict(object_id=object.object_id, display_name=object.display_name, user_principal_name=object.user_principal_name, mail_nickname=object.mail_nickname, mail=object.mail, account_enabled=object.account_enabled, user_type=object.user_type)