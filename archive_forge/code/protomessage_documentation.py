from .message import MessageProtocolEntity
from .proto import ProtoProtocolEntity
from yowsup.layers.protocol_messages.protocolentities.attributes.converter import AttributesConverter
from yowsup.layers.protocol_messages.proto.e2e_pb2 import Message
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
import logging

    <message t="{{TIME_STAMP}}" from="{{CONTACT_JID}}"
        offline="{{OFFLINE}}" type="text" id="{{MESSAGE_ID}}" notify="{{NOTIFY_NAME}}">
            <proto>
                {{SERIALIZE_PROTO_DATA}}
            </proto>
    </message>
    