from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
from ..plugin_utils._hashi_vault_plugin import HashiVaultPlugin
parses a term string into a dictionary