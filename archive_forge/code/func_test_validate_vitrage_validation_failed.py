from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as mistral_client
from heat.engine.resources.openstack.vitrage.vitrage_template import \
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_vitrage_validation_failed(self):
    template = VitrageTemplate('execute_healing', self.resource_def, self.stack)
    self.vitrage.template.validate.return_value = {'results': [{'status': 'validation failed', 'file path': '/tmp/tmpNUEgE3', 'status code': 163, 'message': 'Failed to resolve parameter', 'description': 'Template content validation'}]}
    self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))