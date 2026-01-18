import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_translate_resources_bad_update_policy(self):
    """Test translation of resources including invalid keyword."""
    hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            metadata:\n              foo: bar\n            depends_on: dummy\n            deletion_policy: dummy\n            UpdatePolicy:\n              foo: bar\n        ')
    tmpl = template.Template(hot_tpl)
    err = self.assertRaises(exception.StackValidationFailed, tmpl.__getitem__, tmpl.RESOURCES)
    self.assertEqual('"UpdatePolicy" is not a valid keyword inside a resource definition', str(err))