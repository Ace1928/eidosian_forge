from .notification_groups import GroupsNotificationProtocolEntity
from yowsup.structs import ProtocolTreeNode

<notification notify="{{NOTIFY_NAME}}" id="{{id}}" t="{{TIMESTAMP}}" participant="{{participant_jiid}}" from="{{group_jid}}" type="w:gp2" mode="none">
<remove subject="{{subject}}">
<participant jid="{{participant_jid}}">
</participant>
</remove>
</notification>
    