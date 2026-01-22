from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.lookup.onepassword import OnePass, OnePassCLIv2
from ansible.errors import AnsibleLookupError
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins.lookup import LookupBase
class OnePassCLIv2Doc(OnePassCLIv2):

    def get_raw(self, item_id, vault=None, token=None):
        args = ['document', 'get', item_id]
        if vault is not None:
            args = [*args, '--vault={0}'.format(vault)]
        if self.service_account_token:
            if vault is None:
                raise AnsibleLookupError("'vault' is required with 'service_account_token'")
            environment_update = {'OP_SERVICE_ACCOUNT_TOKEN': self.service_account_token}
            return self._run(args, environment_update=environment_update)
        if token is not None:
            args = [*args, to_bytes('--session=') + token]
        return self._run(args)