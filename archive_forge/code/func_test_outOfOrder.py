import unittest
from .inmemorysenderkeystore import InMemorySenderKeyStore
from ...groups.groupsessionbuilder import GroupSessionBuilder
from ...util.keyhelper import KeyHelper
from ...groups.groupcipher import GroupCipher
from ...duplicatemessagexception import DuplicateMessageException
from ...nosessionexception import NoSessionException
from ...groups.senderkeyname import SenderKeyName
from ...axolotladdress import AxolotlAddress
from ...protocol.senderkeydistributionmessage import SenderKeyDistributionMessage
def test_outOfOrder(self):
    aliceStore = InMemorySenderKeyStore()
    bobStore = InMemorySenderKeyStore()
    aliceSessionBuilder = GroupSessionBuilder(aliceStore)
    bobSessionBuilder = GroupSessionBuilder(bobStore)
    aliceGroupCipher = GroupCipher(aliceStore, 'groupWithBobInIt')
    bobGroupCipher = GroupCipher(bobStore, 'groupWithBobInIt::aliceUserName')
    aliceGroupCipher = GroupCipher(aliceStore, GROUP_SENDER)
    bobGroupCipher = GroupCipher(bobStore, GROUP_SENDER)
    sentAliceDistributionMessage = aliceSessionBuilder.create(GROUP_SENDER)
    receivedAliceDistributionMessage = SenderKeyDistributionMessage(serialized=sentAliceDistributionMessage.serialize())
    bobSessionBuilder.process(GROUP_SENDER, receivedAliceDistributionMessage)
    ciphertexts = []
    for i in range(0, 100):
        ciphertexts.append(aliceGroupCipher.encrypt(b'up the punks'))
    while len(ciphertexts) > 0:
        index = KeyHelper.getRandomSequence(2147483647) % len(ciphertexts)
        ciphertext = ciphertexts.pop(index)
        plaintext = bobGroupCipher.decrypt(ciphertext)
        self.assertEqual(plaintext, b'up the punks')