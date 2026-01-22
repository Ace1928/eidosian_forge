from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _handle_get_csp_enterprise(self):
    """
        Handles the Ansible task when the command is to get the csp enterprise
        """
    self.entity_id = self.parent.enterprise_id
    self.entity = VSPK.NUEnterprise(id=self.entity_id)
    try:
        self.entity.fetch()
    except BambouHTTPError as error:
        self.module.fail_json(msg='Unable to fetch CSP enterprise: {0}'.format(error))
    self.result['id'] = self.entity_id
    self.result['entities'].append(self.entity.to_dict())