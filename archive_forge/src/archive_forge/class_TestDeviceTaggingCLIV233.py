from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestDeviceTaggingCLIV233(TestBlockDeviceTaggingCLIError, TestNICDeviceTaggingCLI):
    """Tests that in microversion 2.33, creating a server with a tagged
    block device will fail, but creating a server with a tagged nic will
    succeed.
    """
    COMPUTE_API_VERSION = '2.33'