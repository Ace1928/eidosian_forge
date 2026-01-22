import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_ports(self):
    exception_list = list()
    for p in self.user_cloud.list_ports():
        if p['name'].startswith(self.new_port_name):
            try:
                self.user_cloud.delete_port(name_or_id=p['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))