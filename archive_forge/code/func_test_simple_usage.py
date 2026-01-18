import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
@mock.patch('novaclient.v2.shell.time')
def test_simple_usage(self, mock_time):
    poll_period = 3
    some_id = 'uuuuuuuuuuuiiiiiiiii'
    updated_objects = (base.Resource(None, info={'not_default_field': 'INPROGRESS'}), base.Resource(None, info={'not_default_field': 'OK'}))
    poll_fn = mock.MagicMock(side_effect=updated_objects)
    novaclient.v2.shell._poll_for_status(poll_fn=poll_fn, obj_id=some_id, status_field='not_default_field', final_ok_states=['ok'], poll_period=poll_period, action='some', silent=True, show_progress=False)
    self.assertEqual([mock.call(poll_period)], mock_time.sleep.call_args_list)
    self.assertEqual([mock.call(some_id)] * 2, poll_fn.call_args_list)