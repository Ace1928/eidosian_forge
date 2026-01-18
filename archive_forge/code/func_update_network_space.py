from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def update_network_space(module, system):
    """
    Update network space.
    Update fields individually. If grouped the API will generate
    a NOT_SUPPORTED_MULTIPLE_UPDATE error.
    """
    space_id = find_network_space_id(module, system)
    datas = [{'interfaces': module.params['interfaces']}, {'mtu': module.params['mtu']}, {'network_config': {'default_gateway': module.params['default_gateway'], 'netmask': module.params['netmask'], 'network': module.params['network']}}, {'rate_limit': module.params['rate_limit']}, {'properties': {'is_async_only': module.params['async_only']}}]
    for data in datas:
        try:
            system.api.put(path=f'network/spaces/{space_id}', data=data)
        except APICommandFailed as err:
            msg = f'Cannot update network space: {err}'
            module.fail_json(msg=msg)
    add_ips_to_network_space(module, system, space_id)
    changed = True
    return changed