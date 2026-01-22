from io import BytesIO
from .... import config, errors, osutils, tests
from .... import transport as _mod_transport
from ... import netrc_credential_store
class TestNetrcCSNoNetrc(tests.TestCaseInTempDir):

    def test_home_netrc_does_not_exist(self):
        self.assertRaises(_mod_transport.NoSuchFile, config.credential_store_registry.get_credential_store, 'netrc')