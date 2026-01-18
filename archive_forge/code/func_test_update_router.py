import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_update_router(self):
    new_router_name = 'mickey'
    new_routes = []
    expected_router_rep = copy.copy(self.mock_router_rep)
    expected_router_rep['name'] = new_router_name
    expected_router_rep['routes'] = new_routes
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers', self.router_id]), json=self.mock_router_rep), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers', self.router_id]), json={'router': expected_router_rep}, validate=dict(json={'router': {'name': new_router_name, 'routes': new_routes}}))])
    new_router = self.cloud.update_router(self.router_id, name=new_router_name, routes=new_routes)
    self._compare_routers(expected_router_rep, new_router)
    self.assert_calls()