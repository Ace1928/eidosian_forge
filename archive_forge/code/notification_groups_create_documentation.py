from .notification_groups import GroupsNotificationProtocolEntity
from yowsup.structs import ProtocolTreeNode

    <notification from="{{owner_username}}-{{group_id}}@g.us" type="w:gp2" id="{{message_id}}" participant="{{participant_jid}}"
            t="{{timestamp}}" notify="{{pushname}}">
        <create type="new" key="{{owner_username}}-{{key}}@temp">
            <group id="{{group_id}}" creator="{{creator_jid}}" creation="{{creation_timestamp}}"
                    subject="{{group_subject}}" s_t="{{subject_timestamp}}" s_o="{{subject_owner_jid}}">
                <participant jid="{{pariticpant_jid}}"/>
                <participant jid="{{}}" type="superadmin"/>
            </group>
        </create>
    </notification>
    