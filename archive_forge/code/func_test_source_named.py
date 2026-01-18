import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
def test_source_named(self):
    self.conf_fixture.config(config_source=['missing_source'])
    with base.mock.patch.object(self.conf, '_open_source_from_opt_group') as open_source:
        self.conf([])
        open_source.assert_called_once_with('missing_source')