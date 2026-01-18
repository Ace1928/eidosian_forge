from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def update_vnic_assignment_order(ucs, vnic, sp):
    from ucsmsdk.mometa.ls.LsVConAssign import LsVConAssign
    mo = LsVConAssign(parent_mo_or_dn=sp, admin_vcon=vnic['admin_vcon'], order=vnic['order'], transport=vnic['transport'], vnic_name=vnic['name'])
    ucs.login_handle.add_mo(mo, True)
    ucs.login_handle.commit()