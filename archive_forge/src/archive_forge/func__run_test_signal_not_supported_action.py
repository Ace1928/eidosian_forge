import datetime
from unittest import mock
from urllib import parse as urlparse
from keystoneauth1 import exceptions as kc_exceptions
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import swift
from heat.engine import scheduler
from heat.engine import stack as stk
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def _run_test_signal_not_supported_action(self, action):
    stack = self._create_stack(TEMPLATE_CFN_SIGNAL)
    rsrc = stack['signal_handler']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    rsrc.action = action
    err_metadata = {'Data': 'foo', 'Status': 'SUCCESS', 'UniqueId': '123'}
    msg = 'Signal resource during %s is not supported.' % action
    exc = self.assertRaises(exception.NotSupported, rsrc.signal, details=err_metadata)
    self.assertEqual(msg, str(exc))