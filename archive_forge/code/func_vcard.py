from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message_meta import MessageMetaAttributes
from .message_media import MediaMessageProtocolEntity
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_contact import ContactAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
@vcard.setter
def vcard(self, value):
    self.media_specific_attributes.vcard = value