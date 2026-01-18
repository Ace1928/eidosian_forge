import collections
import json
import os
from unittest import mock
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources import template_resource
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_template_data_in_create_without_template_file(self):
    self.res.action = self.res.CREATE
    self.res.resource_id = None
    self.res.get_template_file = mock.Mock(side_effect=exception.NotFound(msg_fmt='Could not fetch remote template "test_resource.template": file not found'))
    self.assertRaises(exception.NotFound, self.res.template_data)