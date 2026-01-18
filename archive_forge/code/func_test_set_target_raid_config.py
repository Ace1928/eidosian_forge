import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def test_set_target_raid_config(self):
    self.mgr.set_target_raid_config(NODE1['uuid'], {'fake': 'config'})
    expect = [('PUT', '/v1/nodes/%s/states/raid' % NODE1['uuid'], {}, {'fake': 'config'})]
    self.assertEqual(expect, self.api.calls)