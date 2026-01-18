from __future__ import absolute_import, division, print_function
def update_ip_pool(ucs, module):
    from ucsmsdk.mometa.ippool.IppoolPool import IppoolPool
    mo = IppoolPool(parent_mo_or_dn=module.params['org_dn'], name=module.params['name'], descr=module.params['descr'], assignment_order=module.params['order'])
    ucs.login_handle.add_mo(mo, True)
    ucs.login_handle.commit()
    return mo