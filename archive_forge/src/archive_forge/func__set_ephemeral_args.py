from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def _set_ephemeral_args(args):
    result = []
    if platform.system().lower() == 'sunos':
        result.append('-F')
    else:
        result.append('-t')
    result.append(args['fstype'])
    if args['opts'] != 'defaults':
        result += ['-o', args['opts']]
    result.append(args['src'])
    return result