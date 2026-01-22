from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def append_vgcreate_options(module, state, vgoptions):
    vgcreate_cmd = module.get_bin_path('vgcreate', True)
    autoactivation_supported = is_autoactivation_supported(module=module, vg_cmd=vgcreate_cmd)
    if autoactivation_supported and state in ['active', 'inactive']:
        if VG_AUTOACTIVATION_OPT not in vgoptions:
            if state == 'active':
                vgoptions += [VG_AUTOACTIVATION_OPT, 'y']
            else:
                vgoptions += [VG_AUTOACTIVATION_OPT, 'n']