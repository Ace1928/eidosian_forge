import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class PortManagerTest(testtools.TestCase):

    def setUp(self):
        super(PortManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = ironicclient.v1.port.PortManager(self.api)

    def test_ports_list(self):
        ports = self.mgr.list()
        expect = [('GET', '/v1/ports', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))

    def test_ports_list_by_address(self):
        ports = self.mgr.list(address=PORT['address'])
        expect = [('GET', '/v1/ports/?address=%s' % PORT['address'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))

    def test_ports_list_by_address_detail(self):
        ports = self.mgr.list(address=PORT['address'], detail=True)
        expect = [('GET', '/v1/ports/detail?address=%s' % PORT['address'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))

    def test_ports_list_by_node(self):
        ports = self.mgr.list(node=PORT['node_uuid'])
        expect = [('GET', '/v1/ports/?node=%s' % PORT['node_uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))

    def test_ports_list_by_portgroup(self):
        ports = self.mgr.list(portgroup=PORT['portgroup_uuid'])
        expect = [('GET', '/v1/ports/?portgroup=%s' % PORT['portgroup_uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))

    def test_ports_list_detail(self):
        ports = self.mgr.list(detail=True)
        expect = [('GET', '/v1/ports/detail', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))

    def test_port_list_fields(self):
        ports = self.mgr.list(fields=['uuid', 'address'])
        expect = [('GET', '/v1/ports/?fields=uuid,address', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))

    def test_port_list_detail_and_fields_fail(self):
        self.assertRaises(exc.InvalidAttribute, self.mgr.list, detail=True, fields=['uuid', 'address'])

    def test_ports_list_limit(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.port.PortManager(self.api)
        ports = self.mgr.list(limit=1)
        expect = [('GET', '/v1/ports/?limit=1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(ports, HasLength(1))

    def test_ports_list_marker(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.port.PortManager(self.api)
        ports = self.mgr.list(marker=PORT['uuid'])
        expect = [('GET', '/v1/ports/?marker=%s' % PORT['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(ports, HasLength(1))

    def test_ports_list_pagination_no_limit(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.port.PortManager(self.api)
        ports = self.mgr.list(limit=0)
        expect = [('GET', '/v1/ports', {}, None), ('GET', '/v1/ports/?limit=1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(ports, HasLength(2))

    def test_ports_list_sort_key(self):
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = ironicclient.v1.port.PortManager(self.api)
        ports = self.mgr.list(sort_key='updated_at')
        expect = [('GET', '/v1/ports/?sort_key=updated_at', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(ports))

    def test_ports_list_sort_dir(self):
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = ironicclient.v1.port.PortManager(self.api)
        ports = self.mgr.list(sort_dir='desc')
        expect = [('GET', '/v1/ports/?sort_dir=desc', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(ports))

    def test_ports_show(self):
        port = self.mgr.get(PORT['uuid'])
        expect = [('GET', '/v1/ports/%s' % PORT['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(PORT['uuid'], port.uuid)
        self.assertEqual(PORT['address'], port.address)
        self.assertEqual(PORT['node_uuid'], port.node_uuid)
        self.assertEqual(PORT['pxe_enabled'], port.pxe_enabled)
        self.assertEqual(PORT['local_link_connection'], port.local_link_connection)
        self.assertEqual(PORT['portgroup_uuid'], port.portgroup_uuid)
        self.assertEqual(PORT['physical_network'], port.physical_network)
        self.assertEqual(PORT['is_smartnic'], port.is_smartnic)

    def test_ports_show_by_address(self):
        port = self.mgr.get_by_address(PORT['address'])
        expect = [('GET', '/v1/ports/detail?address=%s' % PORT['address'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(PORT['uuid'], port.uuid)
        self.assertEqual(PORT['address'], port.address)
        self.assertEqual(PORT['node_uuid'], port.node_uuid)
        self.assertEqual(PORT['pxe_enabled'], port.pxe_enabled)
        self.assertEqual(PORT['local_link_connection'], port.local_link_connection)
        self.assertEqual(PORT['portgroup_uuid'], port.portgroup_uuid)
        self.assertEqual(PORT['physical_network'], port.physical_network)
        self.assertEqual(PORT['is_smartnic'], port.is_smartnic)

    def test_port_show_fields(self):
        port = self.mgr.get(PORT['uuid'], fields=['uuid', 'address'])
        expect = [('GET', '/v1/ports/%s?fields=uuid,address' % PORT['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(PORT['uuid'], port.uuid)
        self.assertEqual(PORT['address'], port.address)

    def test_create(self):
        port = self.mgr.create(**CREATE_PORT)
        expect = [('POST', '/v1/ports', {}, CREATE_PORT)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(port)

    def test_create_with_uuid(self):
        port = self.mgr.create(**CREATE_PORT_WITH_UUID)
        expect = [('POST', '/v1/ports', {}, CREATE_PORT_WITH_UUID)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(port)

    def test_delete(self):
        port = self.mgr.delete(port_id=PORT['uuid'])
        expect = [('DELETE', '/v1/ports/%s' % PORT['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(port)

    def test_update(self):
        patch = {'op': 'replace', 'value': NEW_ADDR, 'path': '/address'}
        port = self.mgr.update(port_id=PORT['uuid'], patch=patch)
        expect = [('PATCH', '/v1/ports/%s' % PORT['uuid'], {}, patch)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(NEW_ADDR, port.address)