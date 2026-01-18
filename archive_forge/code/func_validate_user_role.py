from __future__ import absolute_import, division, print_function
import json
import re
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_text
from ansible_collections.ansible.netcommon.plugins.plugin_utils.terminal_base import TerminalBase
def validate_user_role(self):
    user = self._connection._play_context.remote_user
    out = self._exec_cli_command('show user-account %s' % user)
    out = to_text(out, errors='surrogate_then_replace').strip()
    match = re.search('roles:(.+)$', out, re.M)
    if match:
        roles = match.group(1).split()
        if 'network-admin' in roles:
            return True
        return False