from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _flatpak_command(module, noop, command, ignore_failure=False):
    global result
    result['command'] = ' '.join(command)
    if noop:
        result['rc'] = 0
        return ''
    result['rc'], result['stdout'], result['stderr'] = module.run_command(command, check_rc=not ignore_failure)
    return result['stdout']