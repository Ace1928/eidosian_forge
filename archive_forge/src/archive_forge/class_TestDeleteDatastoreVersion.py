from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
class TestDeleteDatastoreVersion(TestDatastores):

    def setUp(self):
        super(TestDeleteDatastoreVersion, self).setUp()
        self.cmd = datastores.DeleteDatastoreVersion(self.app, None)

    def test_delete_datastore_version(self):
        dsversion_id = uuidutils.generate_uuid()
        args = [dsversion_id]
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.dsversion_mgmt_client.delete.assert_called_once_with(dsversion_id)