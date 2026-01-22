from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def activate_vg(module, vg, active):
    changed = False
    vgchange_cmd = module.get_bin_path('vgchange', True)
    vgs_cmd = module.get_bin_path('vgs', True)
    vgs_fields = ['lv_attr']
    autoactivation_enabled = False
    autoactivation_supported = is_autoactivation_supported(module=module, vg_cmd=vgchange_cmd)
    if autoactivation_supported:
        vgs_fields.append('autoactivation')
    vgs_cmd_with_opts = [vgs_cmd, '--noheadings', '-o', ','.join(vgs_fields), '--separator', ';', vg]
    dummy, current_vg_lv_states, dummy = module.run_command(vgs_cmd_with_opts, check_rc=True)
    lv_active_count = 0
    lv_inactive_count = 0
    for line in current_vg_lv_states.splitlines():
        parts = line.strip().split(';')
        if parts[0][4] == 'a':
            lv_active_count += 1
        else:
            lv_inactive_count += 1
        if autoactivation_supported:
            autoactivation_enabled = autoactivation_enabled or parts[1] == 'enabled'
    activate_flag = None
    if active and lv_inactive_count > 0:
        activate_flag = 'y'
    elif not active and lv_active_count > 0:
        activate_flag = 'n'
    if autoactivation_supported:
        if active and (not autoactivation_enabled):
            if module.check_mode:
                changed = True
            else:
                module.run_command([vgchange_cmd, VG_AUTOACTIVATION_OPT, 'y', vg], check_rc=True)
                changed = True
        elif not active and autoactivation_enabled:
            if module.check_mode:
                changed = True
            else:
                module.run_command([vgchange_cmd, VG_AUTOACTIVATION_OPT, 'n', vg], check_rc=True)
                changed = True
    if activate_flag is not None:
        if module.check_mode:
            changed = True
        else:
            module.run_command([vgchange_cmd, '--activate', activate_flag, vg], check_rc=True)
            changed = True
    return changed