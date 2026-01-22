import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class TestRandomString(common.HeatTestCase):
    template_random_string = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  secret1:\n    Type: OS::Heat::RandomString\n  secret2:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 10\n  secret3:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 32\n      character_classes:\n        - class: digits\n          min: 1\n        - class: uppercase\n          min: 1\n        - class: lowercase\n          min: 20\n      character_sequences:\n        - sequence: (),[]{}\n          min: 1\n        - sequence: $_\n          min: 2\n        - sequence: '@'\n          min: 5\n  secret4:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 25\n      character_classes:\n        - class: digits\n          min: 1\n        - class: uppercase\n          min: 1\n        - class: lowercase\n          min: 20\n  secret5:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 10\n      character_sequences:\n        - sequence: (),[]{}\n          min: 1\n        - sequence: $_\n          min: 2\n        - sequence: '@'\n          min: 5\n"

    def create_stack(self, templ):
        self.stack = self.parse_stack(template_format.parse(templ))
        self.assertIsNone(self.stack.create())
        return self.stack

    def parse_stack(self, t):
        stack_name = 'test_stack'
        tmpl = template.Template(t)
        stack = parser.Stack(utils.dummy_context(), stack_name, tmpl)
        stack.validate()
        stack.store()
        return stack

    def assert_min(self, pattern, string, minimum):
        self.assertGreaterEqual(len(re.findall(pattern, string)), minimum)

    def test_random_string(self):
        stack = self.create_stack(self.template_random_string)
        secret1 = stack['secret1']
        random_string = secret1.FnGetAtt('value')
        self.assert_min('[a-zA-Z0-9]', random_string, 32)
        self.assertRaises(exception.InvalidTemplateAttribute, secret1.FnGetAtt, 'foo')
        self.assertEqual(secret1.FnGetRefId(), random_string)
        secret2 = stack['secret2']
        random_string = secret2.FnGetAtt('value')
        self.assert_min('[a-zA-Z0-9]', random_string, 10)
        self.assertEqual(secret2.FnGetRefId(), random_string)
        secret3 = stack['secret3']
        random_string = secret3.FnGetAtt('value')
        self.assertEqual(32, len(random_string))
        self.assert_min('[0-9]', random_string, 1)
        self.assert_min('[A-Z]', random_string, 1)
        self.assert_min('[a-z]', random_string, 20)
        self.assert_min('[(),\\[\\]{}]', random_string, 1)
        self.assert_min('[$_]', random_string, 2)
        self.assert_min('@', random_string, 5)
        self.assertEqual(secret3.FnGetRefId(), random_string)
        secret4 = stack['secret4']
        random_string = secret4.FnGetAtt('value')
        self.assertEqual(25, len(random_string))
        self.assert_min('[0-9]', random_string, 1)
        self.assert_min('[A-Z]', random_string, 1)
        self.assert_min('[a-z]', random_string, 20)
        self.assertEqual(secret4.FnGetRefId(), random_string)
        secret5 = stack['secret5']
        random_string = secret5.FnGetAtt('value')
        self.assertEqual(10, len(random_string))
        self.assert_min('[(),\\[\\]{}]', random_string, 1)
        self.assert_min('[$_]', random_string, 2)
        self.assert_min('@', random_string, 5)
        self.assertEqual(secret5.FnGetRefId(), random_string)
        secret5.resource_id = None
        self.assertEqual('secret5', secret5.FnGetRefId())

    def test_hidden_sequence_property(self):
        hidden_prop_templ = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 100\n      sequence: octdigits\n        "
        stack = self.create_stack(hidden_prop_templ)
        secret = stack['secret']
        random_string = secret.FnGetAtt('value')
        self.assert_min('[0-7]', random_string, 100)
        self.assertEqual(secret.FnGetRefId(), random_string)
        self.assertIsNone(secret.properties['sequence'])
        expected = [{'class': u'octdigits', 'min': 1}]
        self.assertEqual(expected, secret.properties['character_classes'])

    def test_random_string_refid_convergence_cache_data(self):
        t = template_format.parse(self.template_random_string)
        cache_data = {'secret1': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'xyz'})}
        stack = utils.parse_stack(t, cache_data=cache_data)
        rsrc = stack.defn['secret1']
        self.assertEqual('xyz', rsrc.FnGetRefId())

    def test_invalid_length(self):
        template_random_string = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 5\n      character_classes:\n        - class: digits\n          min: 5\n      character_sequences:\n        - sequence: (),[]{}\n          min: 1\n"
        exc = self.assertRaises(exception.StackValidationFailed, self.create_stack, template_random_string)
        self.assertEqual('Length property cannot be smaller than combined character class and character sequence minimums', str(exc))

    def test_max_length(self):
        template_random_string = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 512\n"
        stack = self.create_stack(template_random_string)
        secret = stack['secret']
        random_string = secret.FnGetAtt('value')
        self.assertEqual(512, len(random_string))
        self.assertEqual(secret.FnGetRefId(), random_string)

    def test_exceeds_max_length(self):
        template_random_string = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 513\n"
        exc = self.assertRaises(exception.StackValidationFailed, self.create_stack, template_random_string)
        self.assertIn('513 is out of range (min: 1, max: 512)', str(exc))