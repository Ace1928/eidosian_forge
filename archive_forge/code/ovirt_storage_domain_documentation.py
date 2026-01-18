from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (

        Finds the name of the datacenter that a given
        storage domain is attached to.

        Args:
            sd_name (str): Storage Domain name

        Returns:
            str: Data Center name

        Raises:
            Exception: In case storage domain in not attached to
                an active Datacenter
        