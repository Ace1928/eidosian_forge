from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_service_object(config_data):
    if config_data.get('service_object').get('protocol'):
        commands = []
        for each in config_data.get('service_object').get('protocol'):
            commands.append('service-object {0}'.format(each))
        return commands