from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _load_vg_list(self):
    """Load the VGs from the system."""
    vgs_cmd = self.module.get_bin_path('vgs', required=True)
    vgs_cmd_with_opts = [vgs_cmd, '--noheadings', '--separator', ';', '-o', 'vg_name,vg_uuid']
    dummy, vg_raw_list, dummy = self.module.run_command(vgs_cmd_with_opts, check_rc=True)
    for vg_info in vg_raw_list.splitlines():
        vg_name, vg_uuid = vg_info.strip().split(';')
        self.vg_list.append(vg_name)
        self.vg_list.append(vg_uuid)