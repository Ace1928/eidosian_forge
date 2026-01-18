from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def uninstall_flat(module, binary, names, method):
    """Remove existing flatpaks."""
    global result
    installed_flat_names = [_match_installed_flat_name(module, binary, name, method) for name in names]
    command = [binary, 'uninstall']
    flatpak_version = _flatpak_version(module, binary)
    if LooseVersion(flatpak_version) < LooseVersion('1.1.3'):
        command += ['-y']
    else:
        command += ['--noninteractive']
    command += ['--{0}'.format(method)] + installed_flat_names
    _flatpak_command(module, module.check_mode, command)
    result['changed'] = True