import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_routers(self):
    exception_list = list()
    for router in self.operator_cloud.list_routers():
        if router['name'].startswith(self.router_prefix):
            try:
                self.operator_cloud.delete_router(router['name'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))