from unittest import mock
from mistralclient.auth import keystone
from heat.common import exception
from heat.engine.clients.os import mistral
from heat.tests import common
from heat.tests import utils
class WorkflowConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(WorkflowConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get_workflow_by_identifier = mock.Mock()
        self.ctx.clients.client_plugin('mistral').get_workflow_by_identifier = self.mock_get_workflow_by_identifier
        self.constraint = mistral.WorkflowConstraint()

    def test_validation(self):
        self.mock_get_workflow_by_identifier.return_value = {}
        self.assertTrue(self.constraint.validate('foo', self.ctx))
        self.mock_get_workflow_by_identifier.assert_called_once_with('foo')

    def test_validation_error(self):
        exc = exception.EntityNotFound(entity='Workflow', name='bar')
        self.mock_get_workflow_by_identifier.side_effect = exc
        self.assertFalse(self.constraint.validate('bar', self.ctx))
        self.mock_get_workflow_by_identifier.assert_called_once_with('bar')