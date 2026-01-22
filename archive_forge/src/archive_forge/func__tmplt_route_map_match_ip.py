from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_match_ip(config_data):
    if config_data.get('match') and config_data['match'].get('ip'):

        def construct_cmd_from_list(cmd, config):
            temp = []
            for k, v in iteritems(config):
                temp.append(v)
            cmd += ' ' + ' '.join(sorted(temp))
            return cmd
        cmd = 'match ip'
        if config_data['match']['ip'].get('address'):
            cmd += ' address'
            if config_data['match']['ip']['address'].get('prefix_lists'):
                cmd += ' prefix-list'
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['address']['prefix_lists'])
            elif config_data['match']['ip']['address'].get('acls'):
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['address']['acls'])
        if config_data['match']['ip'].get('flowspec'):
            cmd += ' flowspec'
            if config_data['match']['ip']['flowspec'].get('dest_pfx'):
                cmd += ' dest-pfx'
            elif config_data['match']['ip']['flowspec'].get('src_pfx'):
                cmd += ' src-pfx'
            if config_data['match']['ip']['flowspec'].get('prefix_lists'):
                cmd += ' prefix-list'
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['flowspec']['prefix_lists'])
            elif config_data['match']['ip']['flowspec'].get('acls'):
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['flowspec']['acls'])
        if config_data['match']['ip'].get('next_hop'):
            cmd += ' next-hop'
            if config_data['match']['ip']['next_hop'].get('prefix_lists'):
                cmd += ' prefix-list'
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['next_hop']['prefix_lists'])
            elif config_data['match']['ip']['next_hop'].get('acls'):
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['next_hop']['acls'])
        if config_data['match']['ip'].get('redistribution_source'):
            cmd += ' redistribution-source'
            if config_data['match']['ip']['redistribution_source'].get('prefix_lists'):
                cmd += ' prefix-list'
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['redistribution_source']['prefix_lists'])
            elif config_data['match']['ip']['redistribution_source'].get('acls'):
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['redistribution_source']['acls'])
        if config_data['match']['ip'].get('route_source'):
            cmd += ' route-source'
            if config_data['match']['ip']['route_source'].get('redistribution_source'):
                cmd += ' redistribution-source'
            if config_data['match']['ip']['route_source'].get('prefix_lists'):
                cmd += ' prefix-list'
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['route_source']['prefix_lists'])
            elif config_data['match']['ip']['route_source'].get('acls'):
                cmd = construct_cmd_from_list(cmd, config_data['match']['ip']['route_source']['acls'])
        return cmd