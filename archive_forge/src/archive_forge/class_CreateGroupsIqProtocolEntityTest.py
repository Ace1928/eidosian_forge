from yowsup.layers.protocol_groups.protocolentities.iq_groups_create import CreateGroupsIqProtocolEntity
from yowsup.structs.protocolentity import ProtocolEntityTest
import unittest
class CreateGroupsIqProtocolEntityTest(ProtocolEntityTest, unittest.TestCase):

    def setUp(self):
        self.ProtocolEntity = CreateGroupsIqProtocolEntity
        self.node = entity.toProtocolTreeNode()