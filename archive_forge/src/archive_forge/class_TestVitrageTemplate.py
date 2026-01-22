from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as mistral_client
from heat.engine.resources.openstack.vitrage.vitrage_template import \
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class TestVitrageTemplate(common.HeatTestCase):

    def setUp(self):
        super(TestVitrageTemplate, self).setUp()
        self.ctx = utils.dummy_context()
        template = template_format.parse(vitrage_template)
        self.stack = utils.parse_stack(template, stack_name='test_stack')
        resource_defs = self.stack.t.resource_definitions(self.stack)
        self.resource_def = resource_defs['execute_healing']
        self.vitrage = mock.Mock()
        self.patchobject(VitrageTemplate, 'client', return_value=self.vitrage)
        self.patches = []
        self.patches.append(mock.patch.object(mistral_client, 'mistral_base'))
        self.patches.append(mock.patch.object(mistral_client.MistralClientPlugin, '_create'))
        for patch in self.patches:
            patch.start()
        self.mistral_client = mistral_client.MistralClientPlugin(context=self.ctx)

    def tearDown(self):
        super(TestVitrageTemplate, self).tearDown()
        for patch in self.patches:
            patch.stop()

    def test_create(self):
        template = self._create_resource('execute_healing', self.resource_def, self.stack)
        expected_state = (template.CREATE, template.COMPLETE)
        self.assertEqual(expected_state, template.state)
        self.assertEqual('2fddb683-e32c-4a9b-b8c8-df59af1f5a1a', template.get_reference_id())

    def test_validate(self):
        self.vitrage.template.validate.return_value = {'results': [{'status': 'validation OK', 'file path': '/tmp/tmpNUEgE3', 'message': 'Template validation is OK', 'status code': 0, 'description': 'Template validation'}]}
        template = VitrageTemplate('execute_healing', self.resource_def, self.stack)
        scheduler.TaskRunner(template.validate)()
        self.vitrage.template.validate.assert_called_once()

    def test_validate_vitrage_validate_wrong_format(self):
        """wrong result format for vitrage templete validate"""
        template = VitrageTemplate('execute_healing', self.resource_def, self.stack)
        self.vitrage.template.validate.return_value = {}
        self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))
        self.vitrage.template.validate.return_value = {'results': []}
        self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))
        self.vitrage.template.validate.return_value = {'results': [{'status': 'validation OK', 'file path': '/tmp/tmpNUEgE3', 'message': 'Template validation is OK', 'status code': 0, 'description': 'Template validation'}, {'status': 'validation OK', 'file path': '/tmp/tmpNUEgE3', 'message': 'Template validation is OK', 'status code': 0, 'description': 'Template validation'}]}
        self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))
        self.vitrage.template.validate.return_value = {'results': [{'status': 'validation OK', 'file path': '/tmp/tmpNUEgE3', 'message': 'Template validation is OK', 'description': 'Template validation'}]}
        self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))

    def test_validate_vitrage_validation_failed(self):
        template = VitrageTemplate('execute_healing', self.resource_def, self.stack)
        self.vitrage.template.validate.return_value = {'results': [{'status': 'validation failed', 'file path': '/tmp/tmpNUEgE3', 'status code': 163, 'message': 'Failed to resolve parameter', 'description': 'Template content validation'}]}
        self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))

    def _create_resource(self, name, snippet, stack):
        template = VitrageTemplate(name, snippet, stack)
        self.vitrage.template.add.return_value = [{'status': 'LOADING', 'uuid': '2fddb683-e32c-4a9b-b8c8-df59af1f5a1a', 'status details': 'Template validation is OK', 'date': '2019-02-20 16:36:17.240976', 'type': 'standard', 'name': 'Stack40-execute_healing-4ri7d3vlwp5w'}]
        scheduler.TaskRunner(template.create)()
        return template