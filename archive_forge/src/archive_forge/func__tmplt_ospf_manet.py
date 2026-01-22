from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_manet(config_data):
    if 'manet' in config_data:
        command = []
        if 'cache' in config_data['manet']:
            cmd = 'manet cache'
            if 'acknowledgement' in config_data['manet']['cache']:
                cmd += ' acknowledgement {acknowledgement}'.format(**config_data['manet']['cache'])
            elif 'redundancy' in config_data['manet']['cache']:
                cmd += ' redundancy {redundancy}'.format(**config_data['manet']['cache'])
            command.append(cmd)
        if 'hello' in config_data['manet'] and config_data['manet']['hello']:
            command.append('manet hello')
        if 'peering' in config_data['manet']:
            cmd = 'manet peering selective'
            if 'per_interface' in config_data['manet']['peering']:
                cmd += ' per-interface'
            if 'redundancy' in config_data['manet']['peering']:
                cmd += ' redundancy {redundancy}'.format(**config_data['manet']['peering'])
            command.append(cmd)
        if 'willingness' in config_data['manet']:
            command.append('manet willingness'.format(**config_data['manet']['willingness']))
    return command