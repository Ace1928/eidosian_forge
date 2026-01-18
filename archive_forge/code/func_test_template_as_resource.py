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
def test_template_as_resource(self):
    """Test that resulting resource has the right prop and attrib schema.

        Note that this test requires the Wordpress_Single_Instance.yaml
        template in the templates directory since we want to test using a
        non-trivial template.
        """
    test_templ_name = 'WordPress_Single_Instance.yaml'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates', test_templ_name)
    self.assertIn(test_templ_name, os.listdir(os.path.dirname(path)))
    with open(path) as test_templ_file:
        test_templ = test_templ_file.read()
    self.assertTrue(test_templ, 'Empty test template')
    mock_get = self.patchobject(urlfetch, 'get', return_value=test_templ)
    parsed_test_templ = template_format.parse(test_templ)
    stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template), stack_id=str(uuid.uuid4()))
    properties = {'KeyName': 'mykeyname', 'DBName': 'wordpress1', 'DBUsername': 'wpdbuser', 'DBPassword': 'wpdbpass', 'DBRootPassword': 'wpdbrootpass', 'LinuxDistribution': 'U10'}
    definition = rsrc_defn.ResourceDefinition('test_templ_resource', test_templ_name, properties)
    templ_resource = resource.Resource('test_templ_resource', definition, stack)
    self.assertIsInstance(templ_resource, template_resource.TemplateResource)
    for prop in parsed_test_templ.get('Parameters', {}):
        self.assertIn(prop, templ_resource.properties)
    for attrib in parsed_test_templ.get('Outputs', {}):
        self.assertIn(attrib, templ_resource.attributes)
    for k, v in properties.items():
        self.assertEqual(v, templ_resource.properties[k])
    self.assertEqual({'WordPress_Single_Instance.yaml': 'WordPress_Single_Instance.yaml', 'resources': {}}, stack.env.user_env_as_dict()['resource_registry'])
    self.assertNotIn('WordPress_Single_Instance.yaml', resources.global_env().registry._registry)
    mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('http', 'https'))