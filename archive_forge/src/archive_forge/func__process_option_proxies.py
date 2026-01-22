from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.validation import (
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultOptionGroupBase
def _process_option_proxies(self):
    """check if 'proxies' option is dict or str and set it appropriately"""
    proxies_opt = self._options.get_option('proxies')
    if proxies_opt is None:
        return
    try:
        proxies = check_type_dict(proxies_opt)
    except TypeError:
        proxy = check_type_str(proxies_opt)
        proxies = {'http': proxy, 'https': proxy}
    self._options.set_option('proxies', proxies)