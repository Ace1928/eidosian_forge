from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def transfer_files(module, device):
    dest = module.params['dest']
    recursive = module.params['recursive']
    with SCP(device) as scp:
        for src in module.params['src']:
            if module.params['remote_src']:
                scp.get(src.strip(), local_path=dest, recursive=recursive)
            else:
                scp.put(src.strip(), remote_path=dest, recursive=recursive)