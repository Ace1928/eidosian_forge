import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_roles(self):
    exception_list = list()
    for role in self.operator_cloud.list_roles():
        if role['name'].startswith(self.role_prefix):
            try:
                self.operator_cloud.delete_role(role['name'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))