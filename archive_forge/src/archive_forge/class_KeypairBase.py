import tempfile
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
class KeypairBase(base.TestCase):
    """Methods for functional tests."""

    def keypair_create(self, name=data_utils.rand_uuid()):
        """Create keypair and add cleanup."""
        raw_output = self.openstack('keypair create ' + name)
        self.addCleanup(self.keypair_delete, name, True)
        if not raw_output:
            self.fail('Keypair has not been created!')

    def keypair_list(self, params=''):
        """Return dictionary with list of keypairs."""
        raw_output = self.openstack('keypair list')
        keypairs = self.parse_show_as_object(raw_output)
        return keypairs

    def keypair_delete(self, name, ignore_exceptions=False):
        """Try to delete keypair by name."""
        try:
            self.openstack('keypair delete ' + name)
        except exceptions.CommandFailed:
            if not ignore_exceptions:
                raise