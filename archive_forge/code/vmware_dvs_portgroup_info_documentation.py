from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.six.moves.urllib.parse import unquote

        Return vlan information from given object
        Args:
            vlan_obj: vlan managed object
        Returns: Dict of vlan details of the specific object
        