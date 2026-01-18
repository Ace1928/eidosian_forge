import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def to_actions(dp, acts):
    inst = []
    actions = []
    ofp = dp.ofproto
    parser = dp.ofproto_parser
    for a in acts:
        action = to_action(dp, a)
        if action is not None:
            actions.append(action)
        else:
            action_type = a.get('type')
            if action_type == 'WRITE_ACTIONS':
                write_actions = []
                write_acts = a.get('actions')
                for act in write_acts:
                    action = to_action(dp, act)
                    if action is not None:
                        write_actions.append(action)
                    else:
                        LOG.error('Unknown action type: %s', action_type)
                if write_actions:
                    inst.append(parser.OFPInstructionActions(ofp.OFPIT_WRITE_ACTIONS, write_actions))
            elif action_type == 'CLEAR_ACTIONS':
                inst.append(parser.OFPInstructionActions(ofp.OFPIT_CLEAR_ACTIONS, []))
            elif action_type == 'GOTO_TABLE':
                table_id = UTIL.ofp_table_from_user(a.get('table_id'))
                inst.append(parser.OFPInstructionGotoTable(table_id))
            elif action_type == 'WRITE_METADATA':
                metadata = str_to_int(a.get('metadata'))
                metadata_mask = str_to_int(a['metadata_mask']) if 'metadata_mask' in a else parser.UINT64_MAX
                inst.append(parser.OFPInstructionWriteMetadata(metadata, metadata_mask))
            else:
                LOG.error('Unknown action type: %s', action_type)
    if actions:
        inst.append(parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions))
    return inst