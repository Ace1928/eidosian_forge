from yowsup.layers.protocol_contacts.protocolentities import UpdateContactNotificationProtocolEntity
from yowsup.structs.protocolentity import ProtocolEntityTest
import time
import unittest
class UpdateContactNotificationProtocolEntityTest(ProtocolEntityTest, unittest.TestCase):

    def setUp(self):
        self.ProtocolEntity = UpdateContactNotificationProtocolEntity
        self.node = entity.toProtocolTreeNode()