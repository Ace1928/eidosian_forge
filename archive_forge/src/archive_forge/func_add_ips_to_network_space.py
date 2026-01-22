from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def add_ips_to_network_space(module, system, space_id):
    """ Add IPs to space. Ignore address conflict errors. """
    network_space_name = module.params['name']
    ips = module.params['ips']
    for ip in ips:
        ip_url = f'network/spaces/{space_id}/ips'
        ip_data = ip
        try:
            system.api.post(path=ip_url, data=ip_data)
        except APICommandFailed as err:
            if err.error_code != 'NET_SPACE_ADDRESS_CONFLICT':
                module.fail_json(msg=f'Cannot add IP {ip} to network space {network_space_name}: {err}')