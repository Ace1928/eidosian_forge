from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
import logging

    <iq to="s.whatsapp.net" xmlns="status" type="set" id="{{IQ_ID}}">
        <status>{{MSG}}</status>
    </notification>
    