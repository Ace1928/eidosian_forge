import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_wait_for_drive_state_transition_timeout(self):
    drive = self.driver.ex_list_user_drives()[0]
    state = 'timeout'
    expected_msg = 'Timed out while waiting for drive transition'
    assertRaisesRegex(self, Exception, expected_msg, self.driver._wait_for_drive_state_transition, drive=drive, state=state, timeout=0.5)