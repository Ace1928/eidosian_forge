import logging
import os
from unittest import mock
import fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config_fixture
from oslo_utils import strutils
import testtools
from os_brick.initiator.connectors import nvmeof
def _common_cleanup(self):
    """Runs after each test method to tear down test environment."""
    for x in self.injected:
        try:
            x.stop()
        except AssertionError:
            pass
    for key in [k for k in self.__dict__.keys() if k[0] != '_']:
        del self.__dict__[key]