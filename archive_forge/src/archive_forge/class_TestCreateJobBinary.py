from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
class TestCreateJobBinary(TestJobBinaries):

    def setUp(self):
        super(TestCreateJobBinary, self).setUp()
        self.jb_mock.create.return_value = api_jb.JobBinaries(None, JOB_BINARY_INFO)
        self.jbi_mock = self.app.client_manager.data_processing.job_binary_internals
        self.jbi_mock.create.return_value = mock.Mock(id='jbi_id')
        self.jbi_mock.reset_mock()
        self.cmd = osc_jb.CreateJobBinary(self.app, None)

    def test_job_binary_create_swift_public_protected(self):
        arglist = ['--name', 'job-binary', '--url', 'swift://cont/test', '--username', 'user', '--password', 'pass', '--public', '--protected']
        verifylist = [('name', 'job-binary'), ('url', 'swift://cont/test'), ('username', 'user'), ('password', 'pass'), ('public', True), ('protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.jb_mock.create.assert_called_once_with(description=None, extra={'password': 'pass', 'user': 'user'}, is_protected=True, is_public=True, name='job-binary', url='swift://cont/test')
        expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Name', 'Url')
        self.assertEqual(expected_columns, columns)
        expected_data = ('descr', 'jb_id', False, False, 'job-binary', 'swift://cont/test')
        self.assertEqual(expected_data, data)

    def test_job_binary_create_mutual_exclusion(self):
        arglist = ['job-binary', '--name', 'job-binary', '--access-key', 'ak', '--secret-key', 'sk', '--url', 's3://abc/def', '--password', 'pw']
        with testtools.ExpectedException(osc_u.ParserException):
            self.check_parser(self.cmd, arglist, mock.Mock())