from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
 Creates the ansible result object from a manageiq tenant_quotas entity

        Returns:
            a dict with the applied quotas, name and value
        