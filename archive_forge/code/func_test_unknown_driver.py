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
def test_unknown_driver(self):
    self.conf_fixture.load_raw_values(group='unknown_driver', driver='very_unlikely_to_exist_driver_name')
    source = self.conf._open_source_from_opt_group('unknown_driver')
    self.assertIsNone(source)