from __future__ import absolute_import, division, print_function
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultAuthMethodBase
class HashiVaultAuthMethodNone(HashiVaultAuthMethodBase):
    """HashiVault option group class for auth: none"""
    NAME = 'none'
    OPTIONS = []

    def __init__(self, option_adapter, warning_callback, deprecate_callback):
        super(HashiVaultAuthMethodNone, self).__init__(option_adapter, warning_callback, deprecate_callback)

    def validate(self):
        pass

    def authenticate(self, client, use_token=False):
        return None