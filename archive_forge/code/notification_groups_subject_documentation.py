from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from yowsup.layers.protocol_notifications.protocolentities import NotificationProtocolEntity
from .notification_groups import GroupsNotificationProtocolEntity


    <notification notify="WhatsApp" id="{{id}}" t="{{TIMESTAMP}}" participant="{{PARTICIPANT_JID}}" from="{{GROUP_JID}}" type="w:gp2">
        <subject s_t="{{subject_set_timestamp}}" s_o="{{subject_owner_jid}}" subject="{{SUBJECT}}">
        </subject>
    </notification>

    