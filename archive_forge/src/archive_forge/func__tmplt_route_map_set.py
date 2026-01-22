from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_set(config_data):
    if config_data.get('set'):
        command = []
        set = config_data['set']
        if set.get('aigp_metric'):
            cmd = 'set aigp-metric'
            if set['aigp_metric'].get('value'):
                cmd += ' {value}'.format(**set['aigp_metric'])
            elif set['aigp_metric'].get('igp_metric'):
                cmd += ' igp-metric'
            command.append(cmd)
        if set.get('as_path'):
            cmd = 'set as-path'
            if set['as_path'].get('prepend'):
                cmd += ' prepend'
                if set['as_path']['prepend'].get('as_number'):
                    cmd += ' {0}'.format(set['as_path']['prepend'].get('as_number'))
                elif set['as_path']['prepend'].get('last_as'):
                    cmd += ' last-as {last_as}'.format(**set['as_path']['prepend'])
            if set['as_path'].get('tag'):
                cmd += ' tag'
            command.append(cmd)
        if set.get('automatic_tag'):
            command.append('set automatic-tag')
        if set.get('clns'):
            command.append('set clns next-hop {clns}'.format(**set))
        if set.get('comm_list'):
            command.append('set comm-list {comm_list} delete'.format(**set))
        if set.get('community'):
            cmd = 'set community'
            if set['community'].get('number'):
                cmd += ' ' + ' '.join((i for i in set['community']['number']))
            if set['community'].get('gshut'):
                cmd += ' gshut'
            if set['community'].get('internet'):
                cmd += ' internet'
            if set['community'].get('local_as'):
                cmd += ' local-as'
            if set['community'].get('no_advertise'):
                cmd += ' no-advertise'
            if set['community'].get('no_export'):
                cmd += ' no-export'
            if set['community'].get('none'):
                cmd += ' none'
            if set['community'].get('additive'):
                cmd += ' additive'
            command.append(cmd)
        if set.get('dampening'):
            command.append('set dampening {penalty_half_time} {reuse_route_val} {suppress_route_val} {max_suppress}'.format(**set['dampening']))
        if set.get('default'):
            command.append('set default interface {default}'.format(**set['default']))
        if set.get('extcomm_list'):
            command.append('set extcomm-list {extcomm_list} delete'.format(**set))
        if set.get('extcommunity'):
            if set['extcommunity'].get('cost'):
                cmd = 'set extcommunity cost'
                if set['extcommunity']['cost'].get('igp'):
                    cmd += ' igp'
                elif set['extcommunity']['cost'].get('pre_bestpath'):
                    cmd += ' pre-bestpath'
                if set['extcommunity']['cost'].get('id'):
                    cmd += ' {id}'.format(**set['extcommunity']['cost'])
                if set['extcommunity']['cost'].get('cost_value'):
                    cmd += ' {cost_value}'.format(**set['extcommunity']['cost'])
                command.append(cmd)
            if set['extcommunity'].get('rt'):
                cmd = 'set extcommunity rt'
                if set['extcommunity']['rt'].get('range'):
                    cmd += ' range {lower_limit} {upper_limit}'.format(**set['extcommunity']['rt']['range'])
                elif set['extcommunity']['rt'].get('address'):
                    cmd += ' {address}'.format(**set['extcommunity']['rt'])
                if set['extcommunity']['rt'].get('additive'):
                    cmd += ' additive'
                command.append(cmd)
            if set['extcommunity'].get('soo'):
                command.append('set extcommunity soo {soo}'.format(**set['extcommunity']))
            if set['extcommunity'].get('vpn_distinguisher'):
                cmd = 'set extcommunity vpn-distinguisher'
                if set['extcommunity']['vpn_distinguisher'].get('range'):
                    cmd += ' range {lower_limit} {upper_limit}'.format(**set['extcommunity']['vpn_distinguisher']['range'])
                elif set['extcommunity']['vpn_distinguisher'].get('address'):
                    cmd += ' {address}'.format(**set['extcommunity']['vpn_distinguisher'])
                if set['extcommunity']['vpn_distinguisher'].get('additive'):
                    cmd += ' additive'
                command.append(cmd)
        if set.get('global'):
            command.append('set global')
        if set.get('interfaces'):
            cmd = 'set interface '
            temp = []
            for k, v in iteritems(set['interfaces']):
                temp.append(v)
            cmd += ' '.join(sorted(temp))
            command.append(cmd)
        if set.get('level'):
            cmd = 'set level'
            if set['level'].get('level_1'):
                cmd += ' level-1'
            elif set['level'].get('level_1_2'):
                cmd += ' level-1-2'
            elif set['level'].get('level_2'):
                cmd += ' level-2'
            elif set['level'].get('nssa_only'):
                cmd += ' nssa-only'
        if set.get('lisp'):
            command.append('set lisp locator-set {lisp}'.format(**set))
        if set.get('local_preference'):
            command.append('set local-preference {local_preference}'.format(**set))
        if set.get('metric'):
            cmd = 'set metric'
            if set['metric'].get('metric_value'):
                cmd += ' {metric_value}'.format(**set['metric'])
                if set['metric'].get('deviation'):
                    if set['metric']['deviation'] == 'plus':
                        cmd += ' +{eigrp_delay} {metric_reliability} {metric_bandwidth} {mtu}'.format(**set['metric'])
                    elif set['metric']['deviation'] == 'minus':
                        cmd += ' -{eigrp_delay} {metric_reliability} {metric_bandwidth} {mtu}'.format(**set['metric'])
            if set['metric'].get('deviation') and (not set['metric'].get('eigrp_delay')):
                if set['metric']['deviation'] == 'plus':
                    cmd = 'set metric +{metric_value}'.format(**set['metric'])
                elif set['metric']['deviation'] == 'minus':
                    cmd = 'set metric -{metric_value}'.format(**set['metric'])
            command.append(cmd)
        if set.get('metric_type'):
            cmd = 'set metric-type'
            if set['metric_type'].get('external'):
                cmd += ' external'
            elif set['metric_type'].get('internal'):
                cmd += ' internal'
            elif set['metric_type'].get('type_1'):
                cmd += ' type-1'
            elif set['metric_type'].get('type_2'):
                cmd += ' type-2'
            command.append(cmd)
        if set.get('mpls_label'):
            command.append('set mpls-label')
        if set.get('origin'):
            cmd = 'set origin'
            if set['origin'].get('igp'):
                cmd += ' igp'
            elif set['origin'].get('incomplete'):
                cmd += ' incomplete'
        if set.get('tag'):
            command.append('set tag {tag}'.format(**set))
        if set.get('traffic_index'):
            command.append('set traffic-index {traffic_index}'.format(**set))
        if set.get('vrf'):
            command.append('set vrf {vrf}'.format(**set))
        if set.get('weight'):
            command.append('set weight {weight}'.format(**set))
        return command