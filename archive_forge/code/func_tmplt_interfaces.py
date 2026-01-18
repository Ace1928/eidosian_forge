from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_interfaces(config_data):
    commands = []
    name = config_data.get('name')
    vrf = config_data.get('vrf', '')
    if vrf:
        base_command = 'ntp interface {name} vrf {vrf}'.format(name=name, vrf=vrf)
    else:
        base_command = 'ntp interface {name}'.format(name=name)
    if config_data.get('broadcast_client'):
        commands.append('{base_command} broadcast client'.format(base_command=base_command))
    if config_data.get('broadcast_key'):
        commands.append('{base_command} broadcast key {broadcast_key}'.format(broadcast_key=config_data.get('broadcast_key'), base_command=base_command))
    if config_data.get('broadcast_destination'):
        commands.append('{base_command} broadcast destination {broadcast_destination}'.format(broadcast_destination=config_data.get('broadcast_destination'), base_command=base_command))
    if config_data.get('broadcast_version'):
        commands.append('{base_command} broadcast version {broadcast_version}'.format(broadcast_version=config_data.get('broadcast_version'), base_command=base_command))
    if config_data.get('multicast_destination'):
        commands.append('{base_command} multicast destination {multicast_destination}'.format(multicast_destination=config_data.get('multicast_destination'), base_command=base_command))
    if config_data.get('multicast_client'):
        commands.append('{base_command} multicast client {multicast_client}'.format(multicast_client=config_data.get('multicast_client'), base_command=base_command))
    if config_data.get('multicast_key'):
        commands.append('{base_command} multicast key {multicast_key}'.format(multicast_key=config_data.get('multicast_key'), base_command=base_command))
    elif config_data.get('multicast_version'):
        commands.append('{base_command} multicast version {multicast_version}'.format(multicast_version=config_data.get('multicast_version'), base_command=base_command))
    elif config_data.get('multicast_ttl'):
        commands.append('{base_command} multicast ttl {multicast_ttl}'.format(multicast_ttl=config_data.get('multicast_ttl'), base_command=base_command))
    return commands