from __future__ import absolute_import, division, print_function
import base64
from .vultr_v2 import AnsibleVultr
def transform_result(self, resource):
    if resource:
        resource['user_data'] = self.get_user_data(resource=resource)
    return resource