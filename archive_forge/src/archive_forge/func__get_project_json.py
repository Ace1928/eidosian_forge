from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.lxd import (
from ansible.module_utils.basic import AnsibleModule
import os
def _get_project_json(self):
    return self.client.do('GET', '/1.0/projects/{0}'.format(self.name), ok_error_codes=[404])