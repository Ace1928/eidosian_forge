import pprint
import sys
from testtools import content
from openstack.cloud import meta
from openstack import exceptions
from openstack import proxy
from openstack.tests.functional import base
from openstack import utils
def _cleanup_network(self):
    exception_list = list()
    tb_list = list()
    if self.user_cloud.has_service('network'):
        for r in self.user_cloud.list_routers():
            try:
                if r['name'].startswith(self.new_item_name):
                    self.user_cloud.update_router(r, ext_gateway_net_id=None)
                    for s in self.user_cloud.list_subnets():
                        if s['name'].startswith(self.new_item_name):
                            try:
                                self.user_cloud.remove_router_interface(r, subnet_id=s['id'])
                            except Exception:
                                pass
                    self.user_cloud.delete_router(r.id)
            except Exception as e:
                exception_list.append(e)
                tb_list.append(sys.exc_info()[2])
                continue
        for s in self.user_cloud.list_subnets():
            if s['name'].startswith(self.new_item_name):
                try:
                    self.user_cloud.delete_subnet(s.id)
                except Exception as e:
                    exception_list.append(e)
                    tb_list.append(sys.exc_info()[2])
                    continue
        for n in self.user_cloud.list_networks():
            if n['name'].startswith(self.new_item_name):
                try:
                    self.user_cloud.delete_network(n.id)
                except Exception as e:
                    exception_list.append(e)
                    tb_list.append(sys.exc_info()[2])
                    continue
    if exception_list:
        if len(exception_list) > 1:
            self.addDetail('exceptions', content.text_content('\n'.join([str(ex) for ex in exception_list])))
        exc = exception_list[0]
        raise exc