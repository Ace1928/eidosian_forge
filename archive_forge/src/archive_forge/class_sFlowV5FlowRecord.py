import struct
import logging
class sFlowV5FlowRecord(object):
    _PACK_STR = '!II'
    MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, enterprise, flow_data_format, flow_data_length, flow_data):
        super(sFlowV5FlowRecord, self).__init__()
        self.enterprise = enterprise
        self.flow_data_format = flow_data_format
        self.flow_data_length = flow_data_length
        self.flow_data = flow_data

    @classmethod
    def parser(cls, buf, offset):
        flowdata_format, flow_data_length = struct.unpack_from(cls._PACK_STR, buf, offset)
        format_mask = 4095
        enterprise_shiftbit = 12
        flow_data_format = flowdata_format & format_mask
        enterprise = flowdata_format >> enterprise_shiftbit
        offset += cls.MIN_LEN
        if flow_data_format == 1:
            flow_data = sFlowV5RawPacketHeader.parser(buf, offset)
        elif flow_data_format == 1001:
            flow_data = sFlowV5ExtendedSwitchData.parser(buf, offset)
        else:
            LOG.info('Unknown format. sFlowV5FlowRecord.flow_data_format=%d', flow_data_format)
            pack_str = '!%sc' % flow_data_length
            flow_data = struct.unpack_from(pack_str, buf, offset)
        msg = cls(enterprise, flow_data_format, flow_data_length, flow_data)
        return msg