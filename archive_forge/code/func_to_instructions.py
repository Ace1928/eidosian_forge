import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.lib import ofctl_utils
def to_instructions(dp, insts):
    instructions = []
    ofp = dp.ofproto
    parser = dp.ofproto_parser
    for i in insts:
        inst_type = i.get('type')
        if inst_type in ['APPLY_ACTIONS', 'WRITE_ACTIONS']:
            dics = i.get('actions', [])
            actions = _get_actions(dp, dics)
            if actions:
                if inst_type == 'APPLY_ACTIONS':
                    instructions.append(parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions))
                else:
                    instructions.append(parser.OFPInstructionActions(ofp.OFPIT_WRITE_ACTIONS, actions))
        elif inst_type == 'CLEAR_ACTIONS':
            instructions.append(parser.OFPInstructionActions(ofp.OFPIT_CLEAR_ACTIONS, []))
        elif inst_type == 'GOTO_TABLE':
            table_id = str_to_int(i.get('table_id'))
            instructions.append(parser.OFPInstructionGotoTable(table_id))
        elif inst_type == 'WRITE_METADATA':
            metadata = str_to_int(i.get('metadata'))
            metadata_mask = str_to_int(i['metadata_mask']) if 'metadata_mask' in i else parser.UINT64_MAX
            instructions.append(parser.OFPInstructionWriteMetadata(metadata, metadata_mask))
        elif inst_type == 'METER':
            meter_id = str_to_int(i.get('meter_id'))
            instructions.append(parser.OFPInstructionMeter(meter_id))
        else:
            LOG.error('Unknown instruction type: %s', inst_type)
    return instructions