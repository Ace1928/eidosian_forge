from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def update_ports(module, system):
    """
    Updated mapping of volume to host. If already mapped, exit_json with changed False.
    """
    changed = False
    host = system.hosts.get(name=module.params['host'])
    for wwn_port in module.params['wwns']:
        wwn = WWN(wwn_port)
        if not system.hosts.get_host_by_initiator_address(wwn) == host:
            if not module.check_mode:
                host.add_port(wwn)
            changed = True
    for iscsi_port in module.params['iqns']:
        iscsi_name = make_iscsi_name(iscsi_port)
        if not system.hosts.get_host_by_initiator_address(iscsi_name) == host:
            if not module.check_mode:
                host.add_port(iscsi_name)
            changed = True
    return changed