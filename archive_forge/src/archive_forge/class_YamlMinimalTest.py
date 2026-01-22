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
class YamlMinimalTest(common.HeatTestCase):

    def _parse_template(self, tmpl_str, msg_str):
        parse_ex = self.assertRaises(ValueError, template_format.parse, tmpl_str)
        self.assertIn(msg_str, str(parse_ex))

    def test_long_yaml(self):
        template = {'HeatTemplateFormatVersion': '2012-12-12'}
        config.cfg.CONF.set_override('max_template_size', 10)
        template['Resources'] = ['a'] * int(config.cfg.CONF.max_template_size / 3)
        limit = config.cfg.CONF.max_template_size
        long_yaml = yaml.safe_dump(template)
        self.assertGreater(len(long_yaml), limit)
        ex = self.assertRaises(exception.RequestLimitExceeded, template_format.parse, long_yaml)
        msg = 'Request limit exceeded: Template size (%(actual_len)s bytes) exceeds maximum allowed size (%(limit)s bytes).' % {'actual_len': len(str(long_yaml)), 'limit': config.cfg.CONF.max_template_size}
        self.assertEqual(msg, str(ex))

    def test_parse_no_version_format(self):
        yaml = ''
        self._parse_template(yaml, 'Template format version not found')
        yaml2 = 'Parameters: {}\nMappings: {}\nResources: {}\nOutputs: {}\n'
        self._parse_template(yaml2, 'Template format version not found')

    def test_parse_string_template(self):
        tmpl_str = 'just string'
        msg = 'The template is not a JSON object or YAML mapping.'
        self._parse_template(tmpl_str, msg)

    def test_parse_invalid_yaml_and_json_template(self):
        tmpl_str = '{test'
        msg = 'line 1, column 1'
        self._parse_template(tmpl_str, msg)

    def test_parse_json_document(self):
        tmpl_str = '["foo" , "bar"]'
        msg = 'The template is not a JSON object or YAML mapping.'
        self._parse_template(tmpl_str, msg)

    def test_parse_empty_json_template(self):
        tmpl_str = '{}'
        msg = 'Template format version not found'
        self._parse_template(tmpl_str, msg)

    def test_parse_yaml_template(self):
        tmpl_str = 'heat_template_version: 2013-05-23'
        expected = {'heat_template_version': '2013-05-23'}
        self.assertEqual(expected, template_format.parse(tmpl_str))