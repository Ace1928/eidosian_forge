from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import data_sources as api_ds
from saharaclient.osc.v1 import data_sources as osc_ds
from saharaclient.tests.unit.osc.v1 import test_data_sources as tds_v1
class TestShowDataSource(TestDataSources):

    def setUp(self):
        super(TestShowDataSource, self).setUp()
        self.ds_mock.find_unique.return_value = api_ds.DataSources(None, DS_INFO)
        self.cmd = osc_ds.ShowDataSource(self.app, None)

    def test_data_sources_show(self):
        arglist = ['source']
        verifylist = [('data_source', 'source')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.ds_mock.find_unique.assert_called_once_with(name='source')
        expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Name', 'Type', 'Url')
        self.assertEqual(expected_columns, columns)
        expected_data = ['Data Source for tests', 'id', True, True, 'source', 'swift', 'swift://container.sahara/object']
        self.assertEqual(expected_data, list(data))