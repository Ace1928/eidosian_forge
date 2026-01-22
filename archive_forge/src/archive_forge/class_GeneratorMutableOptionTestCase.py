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
class GeneratorMutableOptionTestCase(base.BaseTestCase):

    def test_include_message(self):
        out = io.StringIO()
        opt = cfg.StrOpt('foo', help='foo option', mutable=True)
        gen = build_formatter(out)
        gen.format(opt, 'group1')
        result = out.getvalue()
        self.assertIn('This option can be changed without restarting.', result)

    def test_do_not_include_message(self):
        out = io.StringIO()
        opt = cfg.StrOpt('foo', help='foo option', mutable=False)
        gen = build_formatter(out)
        gen.format(opt, 'group1')
        result = out.getvalue()
        self.assertNotIn('This option can be changed without restarting.', result)