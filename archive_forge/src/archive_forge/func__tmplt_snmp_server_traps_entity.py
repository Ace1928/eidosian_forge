from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_entity(config_data):
    command = 'snmp-server enable traps entity'
    el = config_data['traps']['entity']
    if el.get('arista_ent_sensor_alarm'):
        command += ' arista-ent-sensor-alarm'
    if el.get('ent_config_change'):
        command += ' ent-config-change'
    if el.get('ent_state_oper'):
        command += ' ent-state-oper'
    if el.get('ent_state_oper_disabled'):
        command += ' ent-state-oper-disabled'
    if el.get('ent_state_oper_enabled'):
        command += ' ent-state-oper-enabled'
    return command