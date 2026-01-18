from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper

        get compute resource object with matching name of esxi_hostname or cluster
        parameters.
        :param recurse: recurse vmware content folder, default is True
        :return: object matching vim.ComputeResource or None if no match
        :rtype: object
        