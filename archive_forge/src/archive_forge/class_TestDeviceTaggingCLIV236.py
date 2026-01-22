from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestDeviceTaggingCLIV236(TestBlockDeviceTaggingCLIError, TestNICDeviceTaggingCLI):
    """Tests that in microversion 2.36, creating a server with a tagged
    block device will fail, but creating a server with a tagged nic will
    succeed. This is testing the boundary before 2.37 where nic tagging
    was broken.
    """
    COMPUTE_API_VERSION = '2.36'