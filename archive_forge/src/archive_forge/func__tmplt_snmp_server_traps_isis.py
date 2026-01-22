from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_isis(config_data):
    command = 'snmp-server enable traps isis'
    el = config_data['traps']['isis']
    if el.get('adjacency_change'):
        command += ' adjacency-change'
    if el.get('area_mismatch'):
        command += ' area-mismatch'
    if el.get('attempt_to_exceed_max_sequence'):
        command += ' attempt-to-exceed-max-sequence'
    if el.get('authentication_type_failure'):
        command += ' authentication-type-failure'
    if el.get('database_overload'):
        command += ' database-overload'
    if el.get('own_lsp_purge'):
        command += ' own-lsp-purge'
    if el.get('rejected_adjacency'):
        command += ' rejected-adjacency'
    if el.get('sequence_number_skip'):
        command += ' sequence-number-skip'
    return command