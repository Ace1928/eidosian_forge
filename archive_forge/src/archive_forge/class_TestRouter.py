from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import router
from openstack.tests.unit import base
class TestRouter(base.TestCase):

    def test_basic(self):
        sot = router.Router()
        self.assertEqual('router', sot.resource_key)
        self.assertEqual('routers', sot.resources_key)
        self.assertEqual('/routers', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = router.Router(**EXAMPLE)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['availability_zone_hints'], sot.availability_zone_hints)
        self.assertEqual(EXAMPLE['availability_zones'], sot.availability_zones)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertTrue(sot.enable_ndp_proxy)
        self.assertFalse(sot.is_distributed)
        self.assertEqual(EXAMPLE['external_gateway_info'], sot.external_gateway_info)
        self.assertEqual(EXAMPLE['flavor_id'], sot.flavor_id)
        self.assertFalse(sot.is_ha)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['revision'], sot.revision_number)
        self.assertEqual(EXAMPLE['routes'], sot.routes)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)

    def test_make_it_with_optional(self):
        sot = router.Router(**EXAMPLE_WITH_OPTIONAL)
        self.assertFalse(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['availability_zone_hints'], sot.availability_zone_hints)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['availability_zones'], sot.availability_zones)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['description'], sot.description)
        self.assertTrue(sot.is_distributed)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['external_gateway_info'], sot.external_gateway_info)
        self.assertTrue(sot.is_ha)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['id'], sot.id)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['name'], sot.name)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['routes'], sot.routes)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['status'], sot.status)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['project_id'], sot.project_id)

    def test_add_interface_subnet(self):
        sot = router.Router(**EXAMPLE)
        response = mock.Mock()
        response.body = {'subnet_id': '3', 'port_id': '2'}
        response.json = mock.Mock(return_value=response.body)
        response.status_code = 200
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        body = {'subnet_id': '3'}
        self.assertEqual(response.body, sot.add_interface(sess, **body))
        url = 'routers/IDENTIFIER/add_router_interface'
        sess.put.assert_called_with(url, json=body)

    def test_add_interface_port(self):
        sot = router.Router(**EXAMPLE)
        response = mock.Mock()
        response.body = {'subnet_id': '3', 'port_id': '3'}
        response.json = mock.Mock(return_value=response.body)
        response.status_code = 200
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        body = {'port_id': '3'}
        self.assertEqual(response.body, sot.add_interface(sess, **body))
        url = 'routers/IDENTIFIER/add_router_interface'
        sess.put.assert_called_with(url, json=body)

    def test_remove_interface_subnet(self):
        sot = router.Router(**EXAMPLE)
        response = mock.Mock()
        response.body = {'subnet_id': '3', 'port_id': '2'}
        response.json = mock.Mock(return_value=response.body)
        response.status_code = 200
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        body = {'subnet_id': '3'}
        self.assertEqual(response.body, sot.remove_interface(sess, **body))
        url = 'routers/IDENTIFIER/remove_router_interface'
        sess.put.assert_called_with(url, json=body)

    def test_remove_interface_port(self):
        sot = router.Router(**EXAMPLE)
        response = mock.Mock()
        response.body = {'subnet_id': '3', 'port_id': '3'}
        response.json = mock.Mock(return_value=response.body)
        response.status_code = 200
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        body = {'network_id': 3, 'enable_snat': True}
        self.assertEqual(response.body, sot.remove_interface(sess, **body))
        url = 'routers/IDENTIFIER/remove_router_interface'
        sess.put.assert_called_with(url, json=body)

    def test_add_interface_4xx(self):
        sot = router.Router(**EXAMPLE)
        response = mock.Mock()
        msg = '.*borked'
        response.body = {'NeutronError': {'message': msg}}
        response.json = mock.Mock(return_value=response.body)
        response.ok = False
        response.status_code = 409
        response.headers = {'content-type': 'application/json'}
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        body = {'subnet_id': '3'}
        with testtools.ExpectedException(exceptions.ConflictException, msg):
            sot.add_interface(sess, **body)

    def test_remove_interface_4xx(self):
        sot = router.Router(**EXAMPLE)
        response = mock.Mock()
        msg = '.*borked'
        response.body = {'NeutronError': {'message': msg}}
        response.json = mock.Mock(return_value=response.body)
        response.ok = False
        response.status_code = 409
        response.headers = {'content-type': 'application/json'}
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        body = {'subnet_id': '3'}
        with testtools.ExpectedException(exceptions.ConflictException, msg):
            sot.remove_interface(sess, **body)

    def test_add_extra_routes(self):
        r = router.Router(**EXAMPLE)
        response = mock.Mock()
        response.headers = {}
        json_body = {'router': {}}
        response.body = json_body
        response.json = mock.Mock(return_value=response.body)
        response.status_code = 200
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        ret = r.add_extra_routes(sess, json_body)
        self.assertIsInstance(ret, router.Router)
        self.assertIsInstance(ret.routes, list)
        url = 'routers/IDENTIFIER/add_extraroutes'
        sess.put.assert_called_with(url, json=json_body)

    def test_remove_extra_routes(self):
        r = router.Router(**EXAMPLE)
        response = mock.Mock()
        response.headers = {}
        json_body = {'router': {}}
        response.body = json_body
        response.json = mock.Mock(return_value=response.body)
        response.status_code = 200
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        ret = r.remove_extra_routes(sess, json_body)
        self.assertIsInstance(ret, router.Router)
        self.assertIsInstance(ret.routes, list)
        url = 'routers/IDENTIFIER/remove_extraroutes'
        sess.put.assert_called_with(url, json=json_body)

    def test_add_router_gateway(self):
        sot = router.Router(**EXAMPLE_WITH_OPTIONAL)
        response = mock.Mock()
        response.body = {'network_id': '3', 'enable_snat': True}
        response.json = mock.Mock(return_value=response.body)
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        body = {'network_id': 3, 'enable_snat': True}
        self.assertEqual(response.body, sot.add_gateway(sess, **body))
        url = 'routers/IDENTIFIER/add_gateway_router'
        sess.put.assert_called_with(url, json=body)

    def test_remove_router_gateway(self):
        sot = router.Router(**EXAMPLE_WITH_OPTIONAL)
        response = mock.Mock()
        response.body = {'network_id': '3', 'enable_snat': True}
        response.json = mock.Mock(return_value=response.body)
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        body = {'network_id': 3, 'enable_snat': True}
        self.assertEqual(response.body, sot.remove_gateway(sess, **body))
        url = 'routers/IDENTIFIER/remove_gateway_router'
        sess.put.assert_called_with(url, json=body)