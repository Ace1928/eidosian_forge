from yowsup.layers.protocol_iq.protocolentities.test_iq import IqProtocolEntityTest
from yowsup.layers.axolotl.protocolentities import SetKeysIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
class SetKeysIqProtocolEntityTest(IqProtocolEntityTest):

    def setUp(self):
        super(SetKeysIqProtocolEntityTest, self).setUp()