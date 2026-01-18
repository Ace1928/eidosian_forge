from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as mistral_client_plugin
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from oslo_serialization import jsonutils
def test_validate_subscriber_type(self):
    t = template_format.parse(subscr_template)
    t['Resources']['MySubscription']['Properties']['subscriber'] = 'foo:ba'
    stack_name = 'test_stack'
    tmpl = template.Template(t)
    self.stack = stack.Stack(self.ctx, stack_name, tmpl)
    exc = self.assertRaises(exception.StackValidationFailed, self.stack.validate)
    self.assertEqual('The subscriber type of must be one of: http, https, mailto, trust+http, trust+https.', str(exc))