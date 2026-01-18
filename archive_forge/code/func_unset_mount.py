from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def unset_mount(module, args):
    """Remove a mount point from fstab."""
    to_write = []
    changed = False
    escaped_name = _escape_fstab(args['name'])
    for line in open(args['fstab'], 'r').readlines():
        if not line.strip():
            to_write.append(line)
            continue
        if line.strip().startswith('#'):
            to_write.append(line)
            continue
        if platform.system() == 'SunOS' and len(line.split()) != 7 or (platform.system() != 'SunOS' and len(line.split()) != 6):
            to_write.append(line)
            continue
        ld = {}
        if platform.system() == 'SunOS':
            ld['src'], dash, ld['name'], ld['fstype'], ld['passno'], ld['boot'], ld['opts'] = line.split()
        else:
            ld['src'], ld['name'], ld['fstype'], ld['opts'], ld['dump'], ld['passno'] = line.split()
        if ld['name'] != escaped_name or ('src' in args and ld['name'] == 'none' and (ld['fstype'] == 'swap') and (ld['src'] != args['src'])):
            to_write.append(line)
            continue
        changed = True
    if changed and (not module.check_mode):
        write_fstab(module, to_write, args['fstab'])
    return (args['name'], changed)