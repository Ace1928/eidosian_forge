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
class AdvancedOptionsTestCase(base.BaseTestCase):
    opts = [cfg.StrOpt('foo', help='foo option', default='fred'), cfg.StrOpt('bar', help='bar option', advanced=True), cfg.StrOpt('foo_bar', help='foobar'), cfg.BoolOpt('bars', help='bars foo', default=True, advanced=True)]

    def test_advanced_option_order_single_ns(self):
        config = [('namespace1', [('alpha', self.opts)])]
        groups = generator._get_groups(config)
        fd, tmp_file = tempfile.mkstemp()
        with open(tmp_file, 'w+') as f:
            formatter = build_formatter(f)
            generator._output_opts(formatter, 'alpha', groups.pop('alpha'))
        expected = '[alpha]\n\n#\n# From namespace1\n#\n\n# foo option (string value)\n#foo = fred\n\n# foobar (string value)\n#foo_bar = <None>\n\n# bar option (string value)\n# Advanced Option: intended for advanced users and not used\n# by the majority of users, and might have a significant\n# effect on stability and/or performance.\n#bar = <None>\n\n# bars foo (boolean value)\n# Advanced Option: intended for advanced users and not used\n# by the majority of users, and might have a significant\n# effect on stability and/or performance.\n#bars = true\n'
        with open(tmp_file, 'r') as f:
            actual = f.read()
        self.assertEqual(expected, actual)