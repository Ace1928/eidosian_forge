from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import (
from ansible_collections.community.hashi_vault.plugins.module_utils._connection_options import HashiVaultConnectionOptions
from ansible_collections.community.hashi_vault.plugins.module_utils._authenticator import HashiVaultAuthenticator
def _generate_retry_callback(self, retry_action):
    """returns a Retry callback function for modules"""

    def _on_retry(retry_obj):
        if retry_obj.total > 0:
            if retry_action == 'warn':
                self.warn('community.hashi_vault: %i %s remaining.' % (retry_obj.total, 'retry' if retry_obj.total == 1 else 'retries'))
            else:
                pass
    return _on_retry