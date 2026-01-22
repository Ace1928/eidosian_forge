from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import node_group_templates as osc_ngt
from saharaclient.tests.unit.osc.v1 import fakes
class TestNodeGroupTemplates(fakes.TestDataProcessing):

    def setUp(self):
        super(TestNodeGroupTemplates, self).setUp()
        self.app.api_version['data_processing'] = '2'
        self.ngt_mock = self.app.client_manager.data_processing.node_group_templates
        self.ngt_mock.reset_mock()