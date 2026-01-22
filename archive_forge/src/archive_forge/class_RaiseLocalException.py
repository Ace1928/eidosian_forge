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
class RaiseLocalException(StackResourceBaseTest):

    def test_heat_exception(self):
        local = exception.StackValidationFailed(message='test')
        self.assertRaises(exception.StackValidationFailed, self.parent_resource.translate_remote_exceptions, local)

    def test_messaging_timeout(self):
        local = msg_exceptions.MessagingTimeout('took too long')
        self.assertRaises(msg_exceptions.MessagingTimeout, self.parent_resource.translate_remote_exceptions, local)

    def test_remote_heat_ex(self):

        class StackValidationFailed_Remote(exception.StackValidationFailed):
            pass
        local = StackValidationFailed_Remote(message='test')
        self.assertRaises(exception.ResourceFailure, self.parent_resource.translate_remote_exceptions, local)