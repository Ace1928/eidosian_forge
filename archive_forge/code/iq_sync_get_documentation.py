from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from .iq_sync import SyncIqProtocolEntity

    <iq type="get" id="{{id}}" xmlns="urn:xmpp:whatsapp:sync">
        <sync mode="{{full | ?}}"
            context="{{registration | ?}}"
            sid="{{str((int(time.time()) + 11644477200) * 10000000)}}"
            index="{{0 | ?}}"
            last="{{true | false?}}"
        >
            <user>
                {{num1}}
            </user>
            <user>
                {{num2}}
            </user>

        </sync>
    </iq>
    