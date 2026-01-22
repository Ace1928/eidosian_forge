from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.validation import (
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultOptionGroupBase
class CallbackRetry(urllib3.util.Retry):

    def __init__(self, *args, **kwargs):
        self._newcb = kwargs.pop('new_callback')
        super(CallbackRetry, self).__init__(*args, **kwargs)

    def new(self, **kwargs):
        if self._newcb is not None:
            self._newcb(self)
        kwargs['new_callback'] = self._newcb
        return super(CallbackRetry, self).new(**kwargs)