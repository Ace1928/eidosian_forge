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
def test_valid_hot_version(self):
    """Test HOT version check.

        Pass a valid HOT version to template.Template.__new__() and
        validate that we get back a parsed template.
        """
    tmpl_str = 'heat_template_version: 2013-05-23'
    hot_tmpl = template_format.parse(tmpl_str)
    parsed_tmpl = template.Template(hot_tmpl)
    expected = ('heat_template_version', '2013-05-23')
    observed = parsed_tmpl.version
    self.assertEqual(expected, observed)