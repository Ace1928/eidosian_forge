from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import jobs as api_j
from saharaclient.osc.v2 import jobs as osc_j
from saharaclient.tests.unit.osc.v1 import test_jobs as tj_v1
class TestDeleteJob(TestJobs):

    def setUp(self):
        super(TestDeleteJob, self).setUp()
        self.j_mock.get.return_value = api_j.Job(None, JOB_INFO)
        self.cmd = osc_j.DeleteJob(self.app, None)

    def test_job_delete(self):
        arglist = ['job_id']
        verifylist = [('job', ['job_id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.j_mock.delete.assert_called_once_with('job_id')