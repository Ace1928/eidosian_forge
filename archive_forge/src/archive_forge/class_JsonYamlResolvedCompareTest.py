import os
from unittest import mock
import re
import yaml
from heat.common import config
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.tests import common
from heat.tests import utils
class JsonYamlResolvedCompareTest(common.HeatTestCase):

    def setUp(self):
        super(JsonYamlResolvedCompareTest, self).setUp()
        self.longMessage = True
        self.maxDiff = None

    def load_template(self, file_name):
        filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates', file_name)
        f = open(filepath)
        t = template_format.parse(f.read())
        f.close()
        return t

    def compare_stacks(self, json_file, yaml_file, parameters):
        t1 = self.load_template(json_file)
        t2 = self.load_template(yaml_file)
        del t1[u'AWSTemplateFormatVersion']
        t1[u'HeatTemplateFormatVersion'] = t2[u'HeatTemplateFormatVersion']
        stack1 = utils.parse_stack(t1, parameters)
        stack2 = utils.parse_stack(t2, parameters)
        t1nr = dict(stack1.t.t)
        del t1nr['Resources']
        t2nr = dict(stack2.t.t)
        del t2nr['Resources']
        self.assertEqual(t1nr, t2nr)
        self.assertEqual(set(stack1), set(stack2))
        for key in stack1:
            self.assertEqual(stack1[key].t, stack2[key].t)

    def test_neutron_resolved(self):
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)
        self.compare_stacks('Neutron.template', 'Neutron.yaml', {})

    def test_wordpress_resolved(self):
        self.compare_stacks('WordPress_Single_Instance.template', 'WordPress_Single_Instance.yaml', {'KeyName': 'test'})