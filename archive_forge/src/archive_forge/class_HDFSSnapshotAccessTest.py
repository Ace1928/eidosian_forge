from tempest.lib import exceptions as tempest_lib_exc
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@testtools.skipUnless(CONF.run_snapshot_tests and CONF.run_mount_snapshot_tests, 'Snapshots or mountable snapshots tests are disabled.')
class HDFSSnapshotAccessTest(SnapshotAccessReadBase):
    protocol = 'hdfs'