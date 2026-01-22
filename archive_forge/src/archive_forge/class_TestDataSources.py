from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import data_sources as api_ds
from saharaclient.osc.v1 import data_sources as osc_ds
from saharaclient.tests.unit.osc.v1 import test_data_sources as tds_v1
class TestDataSources(tds_v1.TestDataSources):

    def setUp(self):
        super(TestDataSources, self).setUp()
        self.app.api_version['data_processing'] = '2'
        self.ds_mock = self.app.client_manager.data_processing.data_sources
        self.ds_mock.reset_mock()