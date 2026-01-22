from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
class TestUpdateJobBinary(TestJobBinaries):

    def setUp(self):
        super(TestUpdateJobBinary, self).setUp()
        self.jb_mock.find_unique.return_value = api_jb.JobBinaries(None, JOB_BINARY_INFO)
        self.jb_mock.update.return_value = api_jb.JobBinaries(None, JOB_BINARY_INFO)
        self.cmd = osc_jb.UpdateJobBinary(self.app, None)

    def test_job_binary_update_all_options(self):
        arglist = ['job-binary', '--name', 'job-binary', '--description', 'descr', '--username', 'user', '--password', 'pass', '--public', '--protected']
        verifylist = [('job_binary', 'job-binary'), ('name', 'job-binary'), ('description', 'descr'), ('username', 'user'), ('password', 'pass'), ('is_public', True), ('is_protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.jb_mock.update.assert_called_once_with('jb_id', {'is_public': True, 'description': 'descr', 'is_protected': True, 'name': 'job-binary', 'extra': {'password': 'pass', 'user': 'user'}})
        expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Name', 'Url')
        self.assertEqual(expected_columns, columns)
        expected_data = ('descr', 'jb_id', False, False, 'job-binary', 'swift://cont/test')
        self.assertEqual(expected_data, data)

    def test_job_binary_update_private_unprotected(self):
        arglist = ['job-binary', '--private', '--unprotected']
        verifylist = [('job_binary', 'job-binary'), ('is_public', False), ('is_protected', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.jb_mock.update.assert_called_once_with('jb_id', {'is_public': False, 'is_protected': False})

    def test_job_binary_update_nothing_updated(self):
        arglist = ['job-binary']
        verifylist = [('job_binary', 'job-binary')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.jb_mock.update.assert_called_once_with('jb_id', {})

    def test_job_binary_update_mutual_exclusion(self):
        arglist = ['job-binary', '--name', 'job-binary', '--access-key', 'ak', '--secret-key', 'sk', '--url', 's3://abc/def', '--password', 'pw']
        with testtools.ExpectedException(osc_u.ParserException):
            self.check_parser(self.cmd, arglist, mock.Mock())