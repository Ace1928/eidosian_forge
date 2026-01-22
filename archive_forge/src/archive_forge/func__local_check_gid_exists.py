from __future__ import absolute_import, division, print_function
import grp
import os
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.sys_info import get_platform_subclass
def _local_check_gid_exists(self):
    if self.gid:
        for gr in grp.getgrall():
            if self.gid == gr.gr_gid and self.name != gr.gr_name:
                self.module.fail_json(msg="GID '{0}' already exists with group '{1}'".format(self.gid, gr.gr_name))