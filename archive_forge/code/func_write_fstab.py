from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def write_fstab(module, lines, path):
    if module.params['backup']:
        backup_file = module.backup_local(path)
    else:
        backup_file = ''
    fs_w = open(path, 'w')
    for l in lines:
        fs_w.write(l)
    fs_w.flush()
    fs_w.close()
    return backup_file