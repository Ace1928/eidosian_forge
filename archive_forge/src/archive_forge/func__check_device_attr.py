from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _check_device_attr(module, device, attr):
    """

    Args:
        module: Ansible module.
        device: device to check attributes.
        attr: attribute to be checked.

    Returns:

    """
    lsattr_cmd = module.get_bin_path('lsattr', True)
    rc, lsattr_out, err = module.run_command(['%s' % lsattr_cmd, '-El', '%s' % device, '-a', '%s' % attr])
    hidden_attrs = ['delalias4', 'delalias6']
    if rc == 255:
        if attr in hidden_attrs:
            current_param = ''
        else:
            current_param = None
        return current_param
    elif rc != 0:
        module.fail_json(msg='Failed to run lsattr: %s' % err, rc=rc, err=err)
    current_param = lsattr_out.split()[1]
    return current_param