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
def _test_output_default_list_opt_with_string_value(self, default):
    opt = cfg.ListOpt('list_opt', help='a list', default=default)
    config = [('namespace1', [('alpha', [opt])])]
    groups = generator._get_groups(config)
    fd, tmp_file = tempfile.mkstemp()
    f = open(tmp_file, 'w+')
    formatter = build_formatter(f)
    expected = '[alpha]\n\n#\n# From namespace1\n#\n\n# a list (list value)\n#list_opt = %(default)s\n' % {'default': default}
    generator._output_opts(formatter, 'alpha', groups.pop('alpha'))
    f.close()
    content = open(tmp_file).read()
    self.assertEqual(expected, content)