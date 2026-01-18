from unittest import mock
from keystoneauth1 import exceptions as keystone_exc
from oslo_config import cfg
import webob
from heat.common import auth_password
from heat.tests import common
Assert that expected environment is present when finally called.