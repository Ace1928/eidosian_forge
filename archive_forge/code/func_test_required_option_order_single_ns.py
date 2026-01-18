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
def test_required_option_order_single_ns(self):
    config = [('namespace1', [('alpha', self.opts)])]
    groups = generator._get_groups(config)
    fd, tmp_file = tempfile.mkstemp()
    with open(tmp_file, 'w+') as f:
        formatter = build_formatter(f, minimal=True)
        generator._output_opts(formatter, 'alpha', groups.pop('alpha'))
    expected = '[alpha]\n\n#\n# From namespace1\n#\n\n# bar option (string value)\nbar = <None>\n\n# bars foo (string value)\nbars = <None>\n'
    with open(tmp_file, 'r') as f:
        actual = f.read()
    self.assertEqual(expected, actual)