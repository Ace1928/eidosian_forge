from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_wait_for_job_error_state(self):
    self._prepare_wait_for_job(constants.JOB_STATE_TERMINATED)
    self.assertRaises(exceptions.WMIJobFailed, self.jobutils._wait_for_job, self._FAKE_JOB_PATH)