import contextlib
import copy
import json as jsonutils
import os
from unittest import mock
from cliff import columns as cliff_columns
import fixtures
from keystoneauth1 import loading
from openstack.config import cloud_region
from openstack.config import defaults
from oslo_utils import importutils
from requests_mock.contrib import fixture
import testtools
from osc_lib import clientmanager
from osc_lib import shell
from osc_lib.tests import fakes
class EnvFixture(fixtures.Fixture):
    """Environment Fixture.

    This fixture replaces os.environ with provided env or an empty env.
    """

    def __init__(self, env=None):
        self.new_env = env or {}

    def _setUp(self):
        self.orig_env, os.environ = (os.environ, self.new_env)
        self.addCleanup(self.revert)

    def revert(self):
        os.environ = self.orig_env