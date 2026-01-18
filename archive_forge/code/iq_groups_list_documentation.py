from yowsup.common import YowConstants
from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from .iq_groups import GroupsIqProtocolEntity

    <iq id="{{id}}"" type="get" to="g.us" xmlns="w:g2">
        <"{{participating | owning}}"></"{{participating | owning}}">
    </iq>
    
    result (processed in iq_result_groups_list.py):
    <iq type="result" from="g.us" id="{{IQ_ID}}">
      <groups>
          <group s_t="{{SUBJECT_TIME}}" creation="{{CREATING_TIME}}" creator="{{OWNER_JID}}" id="{{GROUP_ID}}" s_o="{{SUBJECT_OWNER_JID}}" subject="{{SUBJECT}}">
            <participant jid="{{JID}}" type="admin">
            </participant>
            <participant jid="{{JID}}">
            </participant>
          </group>
          <group s_t="{{SUBJECT_TIME}}" creation="{{CREATING_TIME}}" creator="{{OWNER_JID}}" id="{{GROUP_ID}}" s_o="{{SUBJECT_OWNER_JID}}" subject="{{SUBJECT}}">
            <participant jid="{{JID}}" type="admin">
            </participant>
          </group>
      <groups>
    </iq>
    