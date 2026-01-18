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
def test_stack_create_with_paramfile_validation(self):
    self.register_keystone_auth_fixture()
    self.set_fake_env(FAKE_ENV_KEYSTONE_V2)
    self.shell_error('stack-create teststack --parameter-file private_key=private_key.env --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"', 'Need to specify exactly one of', exception=exc.CommandError)