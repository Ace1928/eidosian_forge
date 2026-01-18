import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_validate_error_reference(self):
    stack_name = 'validate_error_reference'
    tmpl = template_format.parse(main_template)
    files = {'file://tmp/nested.yaml': my_wrong_nested_template}
    stack = parser.Stack(utils.dummy_context(), stack_name, templatem.Template(tmpl, files=files))
    rsrc = stack['volume_server']
    raise_exc_msg = 'InvalidTemplateReference: resources.volume_server<file://tmp/nested.yaml>: The specified reference "instance" (in volume_attachment.Properties.instance_uuid) is incorrect.'
    exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertEqual(raise_exc_msg, str(exc))