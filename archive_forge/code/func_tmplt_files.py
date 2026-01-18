from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_files(config_data):
    command = 'logging'
    if config_data.get('name'):
        command += ' file {name}'.format(name=config_data['name'])
    if config_data.get('path'):
        command += ' path {path}'.format(path=config_data['path'])
    if config_data.get('maxfilesize'):
        command += ' maxfilesize {maxfilesize}'.format(maxfilesize=config_data['maxfilesize'])
    if config_data.get('severity'):
        command += ' severity {severity}'.format(severity=config_data['severity'])
    return command