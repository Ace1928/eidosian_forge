from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib

        Gather information about VM's disks
        Args:
            vm_obj: Managed object of virtual machine
        Returns: A list of dict containing disks information
        