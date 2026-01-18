from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_wait_for_job_ok(self):
    mock_job = self._prepare_wait_for_job(constants.JOB_STATE_COMPLETED_WITH_WARNINGS)
    job = self.jobutils._wait_for_job(self._FAKE_JOB_PATH)
    self.assertEqual(mock_job, job)