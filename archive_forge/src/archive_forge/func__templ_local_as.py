from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _templ_local_as(config_data):
    conf = config_data.get('local_as', {})
    if conf.get('value'):
        command = 'local-as ' + str(conf.get('value', {}))
    if 'no_prepend' in conf:
        command = 'local-as'
        if 'replace_as' in conf.get('no_prepend', {}):
            if 'dual_as' in conf.get('no_prepend', {}).get('replace_as', {}):
                command += ' no-prepend replace-as dual-as'
            elif 'set' in conf.get('no_prepend', {}).get('replace_as', {}):
                command += ' no-prepend replace-as'
        elif 'set' in conf.get('no_prepend', {}):
            command += ' no-prepend'
    return command