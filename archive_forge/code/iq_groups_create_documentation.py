from yowsup.common import YowConstants
from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from .iq_groups import GroupsIqProtocolEntity

    <iq type="set" id="{{id}}" xmlns="w:g2", to="g.us">
        <create subject="{{subject}}">
             <participant jid="{{jid}}"></participant>
        </create>
    </iq>
    