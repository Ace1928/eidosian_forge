from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def write_on_file(content, filename, module):
    path = module.params['path']
    if path[-1] != '/':
        path += '/'
    filepath = '{0}{1}'.format(path, filename)
    try:
        report = open(filepath, 'w')
        report.write(content)
        report.close()
    except Exception:
        module.fail_json(msg='Error while writing on file.')
    return filepath