from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.manila import security_service
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class ManilaSecurityServiceTest(common.HeatTestCase):

    def setUp(self):
        super(ManilaSecurityServiceTest, self).setUp()
        t = template_format.parse(stack_template)
        self.stack = utils.parse_stack(t)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        self.rsrc_defn = resource_defns['security_service']
        self.client = mock.Mock()
        self.patchobject(security_service.SecurityService, 'client', return_value=self.client)

    def _create_resource(self, name, snippet, stack):
        ss = security_service.SecurityService(name, snippet, stack)
        value = mock.MagicMock(id='12345')
        self.client.security_services.create.return_value = value
        self.client.security_services.get.return_value = value
        scheduler.TaskRunner(ss.create)()
        args = self.client.security_services.create.call_args[1]
        self.assertEqual(self.rsrc_defn._properties, args)
        self.assertEqual('12345', ss.resource_id)
        return ss

    def test_create(self):
        ct = self._create_resource('security_service', self.rsrc_defn, self.stack)
        expected_state = (ct.CREATE, ct.COMPLETE)
        self.assertEqual(expected_state, ct.state)
        self.assertEqual('security_services', ct.entity)

    def test_create_failed(self):
        ss = security_service.SecurityService('security_service', self.rsrc_defn, self.stack)
        self.client.security_services.create.side_effect = Exception('error')
        exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(ss.create))
        expected_state = (ss.CREATE, ss.FAILED)
        self.assertEqual(expected_state, ss.state)
        self.assertIn('Exception: resources.security_service: error', str(exc))

    def test_update(self):
        ss = self._create_resource('security_service', self.rsrc_defn, self.stack)
        t = template_format.parse(stack_template_update)
        rsrc_defns = template.Template(t).resource_definitions(self.stack)
        new_ss = rsrc_defns['security_service']
        scheduler.TaskRunner(ss.update, new_ss)()
        args = {'domain': 'fake-domain', 'password': None, 'user': None, 'server': 'fake-server', 'name': 'fake_security_service'}
        self.client.security_services.update.assert_called_once_with('12345', **args)
        self.assertEqual((ss.UPDATE, ss.COMPLETE), ss.state)

    def test_update_replace(self):
        ss = self._create_resource('security_service', self.rsrc_defn, self.stack)
        t = template_format.parse(stack_template_update_replace)
        rsrc_defns = template.Template(t).resource_definitions(self.stack)
        new_ss = rsrc_defns['security_service']
        self.assertEqual(0, self.client.security_services.update.call_count)
        err = self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(ss.update, new_ss))
        msg = 'The Resource security_service requires replacement.'
        self.assertEqual(msg, str(err))