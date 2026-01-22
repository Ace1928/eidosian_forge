import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import clusters
class ClusterStatusTest(testtools.TestCase):

    def test_constants(self):
        self.assertEqual('ACTIVE', clusters.ClusterStatus.ACTIVE)
        self.assertEqual('BUILD', clusters.ClusterStatus.BUILD)
        self.assertEqual('FAILED', clusters.ClusterStatus.FAILED)
        self.assertEqual('SHUTDOWN', clusters.ClusterStatus.SHUTDOWN)