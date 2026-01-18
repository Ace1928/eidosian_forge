from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_lag(module, blade):
    """Update Link Aggregation Group"""
    changed = False
    used_ports = []
    current_ports = []
    lagfact = []
    current_lag = list(blade.get_link_aggregation_groups(names=[module.params['name']]).items)[0]
    for port in range(0, len(current_lag.ports)):
        used_ports.append(current_lag.ports[port].name)
    for lag_port in range(0, len(module.params['ports']), 2):
        if not module.params['ports'][lag_port].split('.')[0].upper() + '.FM1.' + module.params['ports'][lag_port].split('.')[1].upper() in used_ports:
            current_lags = list(blade.get_link_aggregation_groups().items)
            for lag in range(0, len(current_lags)):
                for port in range(0, len(current_lags[lag].ports)):
                    current_ports.append(current_lags[lag].ports[port].name)
            for current_lag_port in range(0, len(current_ports)):
                if module.params['ports'][lag_port].split('.')[0].upper() + '.FM1.' + module.params['ports'][lag_port].split('.')[1].upper() in current_ports:
                    module.fail_json(msg='Selected port {0} is currently in use by another LAG.'.format(module.params['ports'][lag_port].upper()))
    new_ports = []
    for port in range(0, len(module.params['ports'])):
        if module.params['ports'][port].split('.')[0].upper()[0] != 'X':
            new_ports.append(module.params['ports'][port].split('.')[0].upper() + '.FM1.' + module.params['ports'][port].split('.')[1].upper())
            new_ports.append(module.params['ports'][port].split('.')[0].upper() + '.FM2.' + module.params['ports'][port].split('.')[1].upper())
        else:
            new_ports.append(module.params['ports'][port].upper())
    ports = []
    for final_port in range(0, len(new_ports)):
        ports.append(flashblade.FixedReference(name=new_ports[final_port]))
    link_aggregation_group = flashblade.Linkaggregationgroup(ports=ports)
    if sorted(used_ports) != sorted(new_ports):
        changed = True
        if not module.check_mode:
            res = blade.patch_link_aggregation_groups(names=[module.params['name']], link_aggregation_group=link_aggregation_group)
            if res.status_code != 200:
                module.fail_json(msg='Failed to update LAG {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
            else:
                response = list(res.items)[0]
                lagfact = {'mac_address': response.mac_address, 'port_speed': str(response.port_speed / 1000000000) + 'Gb/s', 'lag_speed': str(response.lag_speed / 1000000000) + 'Gb/s', 'status': response.status}
    module.exit_json(changed=changed, lag=lagfact)