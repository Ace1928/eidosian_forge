import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_groups(self):
    exception_list = list()
    for group in self.operator_cloud.list_groups():
        if group['name'].startswith(self.group_prefix):
            try:
                self.operator_cloud.delete_group(group['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))