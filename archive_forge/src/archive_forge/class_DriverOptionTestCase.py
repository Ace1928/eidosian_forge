import io
import sys
import textwrap
from unittest import mock
import fixtures
from oslotest import base
import tempfile
import testscenarios
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_config import generator
from oslo_config import types
import yaml
class DriverOptionTestCase(base.BaseTestCase):

    def setUp(self):
        super(DriverOptionTestCase, self).setUp()
        self.conf = cfg.ConfigOpts()
        self.config_fixture = config_fixture.Config(self.conf)
        self.config = self.config_fixture.config
        self.useFixture(self.config_fixture)

    @mock.patch.object(generator, '_get_driver_opts_loaders')
    @mock.patch.object(generator, '_get_raw_opts_loaders')
    @mock.patch.object(generator, 'LOG')
    def test_driver_option(self, mock_log, raw_opts_loader, driver_opts_loader):
        group = cfg.OptGroup(name='test_group', title='Test Group', driver_option='foo')
        regular_opts = [cfg.MultiStrOpt('foo', help='foo option'), cfg.StrOpt('bar', help='bar option')]
        driver_opts = {'d1': [cfg.StrOpt('d1-foo', help='foo option')], 'd2': [cfg.StrOpt('d2-foo', help='foo option')]}
        raw_opts_loader.return_value = [('testing', lambda: [(group, regular_opts)])]
        driver_opts_loader.return_value = [('testing', lambda: driver_opts)]
        generator.register_cli_opts(self.conf)
        self.config(namespace=['test_generator'], format_='yaml')
        stdout = io.StringIO()
        generator.generate(self.conf, output_file=stdout)
        body = stdout.getvalue()
        actual = yaml.safe_load(body)
        test_section = actual['options']['test_group']
        self.assertEqual('foo', test_section['driver_option'])
        found_option_names = [o['name'] for o in test_section['opts']]
        self.assertEqual(['foo', 'bar', 'd1-foo', 'd2-foo'], found_option_names)
        self.assertEqual({'d1': ['d1-foo'], 'd2': ['d2-foo']}, test_section['driver_opts'])