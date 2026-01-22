from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.vultr_v2 import AnsibleVultr, vultr_argument_spec
class AnsibleVultrStartupScript(AnsibleVultr):

    def configure(self):
        if self.module.params['script']:
            self.module.params['script'] = base64.b64encode(self.module.params['script'].encode())

    def update(self, resource):
        resource['script'] = resource['script'].encode()
        return super(AnsibleVultrStartupScript, self).update(resource=resource)

    def transform_result(self, resource):
        if resource:
            resource['script'] = base64.b64decode(resource['script']).decode()
        return resource