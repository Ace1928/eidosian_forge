from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import properties
from heat.engine import resource
from heat.scaling import lbutils
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
class MockedAWSLB(generic_resource.GenericResource):
    properties_schema = {'Instances': properties.Schema(properties.Schema.LIST, update_allowed=True)}

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        return