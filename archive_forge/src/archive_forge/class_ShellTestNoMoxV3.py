import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
class ShellTestNoMoxV3(ShellTestNoMox):

    def _set_fake_env(self):
        fake_env_kwargs = {'OS_NO_CLIENT_AUTH': 'True', 'HEAT_URL': 'http://heat.example.com'}
        fake_env_kwargs.update(FAKE_ENV_KEYSTONE_V3)
        self.set_fake_env(fake_env_kwargs)