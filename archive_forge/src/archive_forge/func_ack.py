from yowsup.structs import ProtocolEntity
from yowsup.layers.protocol_receipts.protocolentities  import OutgoingReceiptProtocolEntity
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message_meta import MessageMetaAttributes
from copy import deepcopy
def ack(self, read=False):
    return OutgoingReceiptProtocolEntity(self.getId(), self.getFrom(), read, participant=self.getParticipant())