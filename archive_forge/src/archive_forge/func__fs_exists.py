from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils._mount import ismount
import re
def _fs_exists(module, filesystem):
    """
    Check if file system already exists on /etc/filesystems.

    :param module: Ansible module.
    :param community.general.filesystem: filesystem name.
    :return: True or False.
    """
    lsfs_cmd = module.get_bin_path('lsfs', True)
    rc, lsfs_out, err = module.run_command([lsfs_cmd, '-l', filesystem])
    if rc == 1:
        if re.findall('No record matching', err):
            return False
        else:
            module.fail_json(msg='Failed to run lsfs. Error message: %s' % err)
    else:
        return True