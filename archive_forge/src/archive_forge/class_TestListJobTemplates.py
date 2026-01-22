from unittest import mock
from saharaclient.api import job_types as api_jt
from saharaclient.api.v2 import job_templates as api_job_templates
from saharaclient.osc.v2 import job_types as osc_jt
from saharaclient.tests.unit.osc.v1 import test_job_types as tjt_v1
class TestListJobTemplates(TestJobTypes):

    def setUp(self):
        super(TestListJobTemplates, self).setUp()
        self.jt_mock.list.return_value = [api_jt.JobType(None, JOB_TYPE_INFO)]
        self.cmd = osc_jt.ListJobTypes(self.app, None)

    def test_job_types_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['Name', 'Plugins']
        self.assertEqual(expected_columns, columns)
        expected_data = [('Pig', 'fake(0.1, 0.2), wod(6.2.2)')]
        self.assertEqual(expected_data, list(data))

    def test_job_types_list_extra_search_opts(self):
        arglist = ['--type', 'Pig', '--plugin', 'fake', '--plugin-version', '0.1']
        verifylist = [('type', 'Pig'), ('plugin', 'fake'), ('plugin_version', '0.1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['Name', 'Plugins']
        self.assertEqual(expected_columns, columns)
        expected_data = [('Pig', 'fake(0.1, 0.2), wod(6.2.2)')]
        self.assertEqual(expected_data, list(data))