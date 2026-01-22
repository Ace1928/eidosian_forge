from yowsup.layers.protocol_contacts.protocolentities.iq_sync_result import ResultSyncIqProtocolEntity
from yowsup.structs.protocolentity import ProtocolEntityTest
import unittest
class ResultSyncIqProtocolEntityTest(ProtocolEntityTest, unittest.TestCase):

    def setUp(self):
        self.ProtocolEntity = ResultSyncIqProtocolEntity
        self.node = entity.toProtocolTreeNode()

    def test_delta_result(self):
        del self.node.getChild('sync')['wait']
        self.test_generation()