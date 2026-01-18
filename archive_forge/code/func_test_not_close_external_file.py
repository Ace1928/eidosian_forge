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
@mock.patch.object(generator, '_get_groups')
@mock.patch.object(generator, '_list_opts')
def test_not_close_external_file(self, a, b):
    generator.register_cli_opts(self.conf)
    self.config()
    m = mock.Mock()
    generator.generate(self.conf, output_file=m)
    m().close.assert_not_called()