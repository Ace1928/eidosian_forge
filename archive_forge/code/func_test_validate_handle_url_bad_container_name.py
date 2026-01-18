import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from swiftclient import client as swiftclient_client
from swiftclient import exceptions as swiftclient_exceptions
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import swift
from heat.engine import node_data
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template as templatem
from heat.tests import common
from heat.tests import utils
@mock.patch.object(swift.SwiftClientPlugin, 'get_signal_url')
def test_validate_handle_url_bad_container_name(self, mock_handle_url):
    mock_handle_url.return_value = 'http://fake-host.com:8080/v1/AUTH_test_tenant/my-container/test_st-test_wait_condition_handle?temp_url_sig=12d8f9f2c923fbeb555041d4ed63d83de6768e95&temp_url_expires=1404762741'
    st = create_stack(swiftsignal_template)
    st.create()
    self.assertIn('not a valid SwiftSignalHandle.  The container name', str(st.status_reason))