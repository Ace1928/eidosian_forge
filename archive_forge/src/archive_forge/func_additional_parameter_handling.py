from __future__ import absolute_import, division, print_function
import errno
import os
import shutil
import sys
import time
from pwd import getpwnam, getpwuid
from grp import getgrnam, getgrgid
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def additional_parameter_handling(params):
    """Additional parameter validation and reformatting"""
    if params['state'] not in ('link', 'absent') and os.path.isdir(to_bytes(params['path'], errors='surrogate_or_strict')):
        basename = None
        if params['_original_basename']:
            basename = params['_original_basename']
        elif params['src']:
            basename = os.path.basename(params['src'])
        if basename:
            params['path'] = os.path.join(params['path'], basename)
    prev_state = get_state(to_bytes(params['path'], errors='surrogate_or_strict'))
    if params['state'] is None:
        if prev_state != 'absent':
            params['state'] = prev_state
        elif params['recurse']:
            params['state'] = 'directory'
        else:
            params['state'] = 'file'
    if params['recurse'] and params['state'] != 'directory':
        raise ParameterError(results={'msg': "recurse option requires state to be 'directory'", 'path': params['path']})
    if params['src'] and params['state'] not in ('link', 'hard'):
        raise ParameterError(results={'msg': "src option requires state to be 'link' or 'hard'", 'path': params['path']})