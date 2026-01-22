from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_match(config_data):
    if config_data.get('match') and (not config_data['match'].get('ip')) and (not config_data['match'].get('ipv6')):
        command = []
        match = config_data['match']
        if match and match.get('additional_paths'):
            cmd = 'match additional-paths advertise-set'
            if config_data['match']['additional_paths'].get('all'):
                cmd += ' all'
            if config_data['match']['additional_paths'].get('best'):
                cmd += ' best {best}'.format(**config_data['match']['additional_paths'])
            if config_data['match']['additional_paths'].get('best_range'):
                cmd += ' best-range'
                if config_data['match']['additional_paths']['best_range'].get('lower_limit'):
                    cmd += ' lower-limit {lower_limit}'.format(**config_data['match']['additional_paths']['best_range'])
                if config_data['match']['additional_paths']['best_range'].get('upper_limit'):
                    cmd += ' upper-limit {upper_limit}'.format(**config_data['match']['additional_paths']['best_range'])
            if config_data['match']['additional_paths'].get('group_best'):
                cmd += ' group-best'
            command.append(cmd)
        if match.get('as_path'):
            cmd = 'match as-path '
            if match['as_path'].get('acls'):
                temp = []
                for k, v in iteritems(match['as_path']['acls']):
                    temp.append(str(v))
                cmd += ' '.join(sorted(temp))
            command.append(cmd)
        if match.get('clns'):
            cmd = 'match clns'
            if match['clns'].get('address'):
                cmd += ' address {address}'.format(**match['clns'])
            elif match['clns'].get('next_hop'):
                cmd = ' next-hop {next_hop}'.format(**match['clns'])
            elif match['clns'].get('route_source'):
                cmd = ' route-source {route_source}'.format(**match['clns'])
            command.append(cmd)
        if match.get('community'):
            cmd = 'match community '
            temp = []
            for k, v in iteritems(match['community']['name']):
                temp.append(v)
            cmd += ' '.join(sorted(temp))
            if match['community'].get('exact_match'):
                cmd += ' exact-match'
            command.append(cmd)
        if match.get('extcommunity'):
            cmd = 'match extcommunity '
            temp = []
            for k, v in iteritems(match['extcommunity']):
                temp.append(v)
            cmd += ' '.join(sorted(temp))
            command.append(cmd)
        if match.get('interfaces'):
            cmd = 'match interface '
            temp = []
            for k, v in iteritems(match['interfaces']):
                temp.append(v)
            cmd += ' '.join(sorted(temp))
            command.append(cmd)
        if match.get('length'):
            command.append('match length {minimum} {maximum}'.format(**match['length']))
        if match.get('local_preference'):
            cmd = 'match local-preference '
            if match['local_preference'].get('value'):
                temp = []
                for k, v in iteritems(match['local_preference']['value']):
                    temp.append(v)
                cmd += ' '.join(sorted(temp))
            command.append(cmd)
        if match.get('mdt_group'):
            cmd = 'match mdt-group '
            if match['mdt_group'].get('acls'):
                temp = []
                for k, v in iteritems(match['mdt_group']['acls']):
                    temp.append(v)
                cmd += ' '.join(sorted(temp))
            command.append(cmd)
        if match.get('metric'):
            cmd = 'match metric'
            if match['metric'].get('external'):
                cmd += ' external'
            if match['metric'].get('value'):
                cmd += ' {value}'.format(**match['metric'])
            if match['metric'].get('deviation'):
                cmd += ' +-'
                if match['metric'].get('deviation_value'):
                    cmd += ' {deviation_value}'.format(**match['metric'])
            command.append(cmd)
        if match.get('mpls_label'):
            command.append('match mpls-label')
        if match.get('policy_lists'):
            cmd = 'match policy-list '
            temp = []
            for k, v in iteritems(match['policy_lists']):
                temp.append(v)
            cmd += ' '.join(sorted(temp))
            command.append(cmd)
        if match.get('route_type'):
            cmd = 'match route-type'
            if match['route_type'].get('external'):
                cmd += ' external'
                if match['route_type']['external'].get('type_1'):
                    cmd += ' type-1'
                elif match['route_type']['external'].get('type_2'):
                    cmd += ' type-2'
            elif match['route_type'].get('internal'):
                cmd += ' internal'
            elif match['route_type'].get('level_1'):
                cmd += ' level-1'
            elif match['route_type'].get('level_2'):
                cmd += ' level-2'
            elif match['route_type'].get('local'):
                cmd += ' local'
            elif match['route_type'].get('nssa_external'):
                cmd += ' nssa-external'
                if match['route_type']['nssa_external'].get('type_1'):
                    cmd += ' type-1'
                elif match['route_type']['nssa_external'].get('type_2'):
                    cmd += ' type-2'
            command.append(cmd)
        if match.get('rpki'):
            cmd = 'match rpki'
            if match['rpki'].get('invalid'):
                cmd += ' invalid'
            if match['rpki'].get('not_found'):
                cmd += ' not-found'
            if match['rpki'].get('valid'):
                cmd += ' valid'
            command.append(cmd)
        if match.get('security_group'):
            cmd = 'match security-group'
            if match['security_group'].get('source'):
                cmd += ' source tag '
                temp = []
                for k, v in iteritems(match['security_group']['source']):
                    temp.append(str(v))
                cmd += ' '.join(sorted(temp))
            elif match['security_group'].get('destination'):
                cmd += ' destination tag'
                for each in match['destination']:
                    cmd += ' {0}'.format(each)
            command.append(cmd)
        if match.get('source_protocol'):
            cmd = 'match source-protocol'
            if match['source_protocol'].get('bgp'):
                cmd += ' bgp {bgp}'.format(**match['source_protocol'])
            if match['source_protocol'].get('connected'):
                cmd += ' connected'
            if match['source_protocol'].get('eigrp'):
                cmd += ' eigrp {eigrp}'.format(**match['source_protocol'])
            if match['source_protocol'].get('isis'):
                cmd += ' isis'
            if match['source_protocol'].get('lisp'):
                cmd += ' lisp'
            if match['source_protocol'].get('mobile'):
                cmd += ' mobile'
            if match['source_protocol'].get('ospf'):
                cmd += ' ospf {ospf}'.format(**match['source_protocol'])
            if match['source_protocol'].get('ospfv3'):
                cmd += ' ospfv3 {ospfv3}'.format(**match['source_protocol'])
            if match['source_protocol'].get('rip'):
                cmd += ' rip'
            if match['source_protocol'].get('static'):
                cmd += ' static'
            command.append(cmd)
        if match.get('tag'):
            cmd = 'match tag'
            if match['tag'].get('tag_list'):
                cmd += ' list'
                for each in match['tag']['tag_list']:
                    cmd += ' {0}'.format(each)
            elif match['tag'].get('value'):
                for each in match['tag']['value']:
                    cmd += ' {0}'.format(each)
            command.append(cmd)
        if match.get('track'):
            command.append('match track {track}'.format(**match))
        return command