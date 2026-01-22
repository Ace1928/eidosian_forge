from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_networks(self):
    exception_list = list()
    for network in self.operator_cloud.list_networks():
        if network['name'].startswith(self.network_name):
            try:
                self.operator_cloud.delete_network(network['name'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))