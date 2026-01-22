from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
class TestDatastoreList(TestDatastores):
    columns = datastores.ListDatastores.columns
    values = ('d-123', 'mysql')

    def setUp(self):
        super(TestDatastoreList, self).setUp()
        self.cmd = datastores.ListDatastores(self.app, None)
        data = [self.fake_datastores.get_datastores_d_123()]
        self.datastore_client.list.return_value = common.Paginated(data)

    def test_datastore_list_defaults(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.datastore_client.list.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual([self.values], data)