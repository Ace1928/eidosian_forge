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
def test_wait_for_provision_state_wrong_input(self):
    self.assertRaises(ValueError, self.mgr.wait_for_provision_state, 'node', 'active', timeout='42')
    self.assertRaises(ValueError, self.mgr.wait_for_provision_state, 'node', 'active', timeout=-1)
    self.assertRaises(TypeError, self.mgr.wait_for_provision_state, 'node', 'active', poll_delay_function={})