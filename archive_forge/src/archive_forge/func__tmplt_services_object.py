from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_services_object(config_data):
    if config_data.get('services_object'):
        cmd = 'service-object {protocol}'.format(**config_data['services_object'])
        if config_data['services_object'].get('source_port'):
            if config_data['services_object']['source_port'].get('range'):
                cmd += ' source range {start} {end}'.format(**config_data['services_object']['source_port']['range'])
            else:
                key = list(config_data['services_object']['source_port'])[0]
                cmd += ' source {0} {1}'.format(key, config_data['services_object']['source_port'][key])
        if config_data['services_object'].get('destination_port'):
            if config_data['services_object']['destination_port'].get('range'):
                cmd += ' destination range {start} {end}'.format(**config_data['services_object']['destination_port']['range'])
            else:
                key = list(config_data['services_object']['destination_port'])[0]
                cmd += ' destination {0} {1}'.format(key, config_data['services_object']['destination_port'][key])
        return cmd