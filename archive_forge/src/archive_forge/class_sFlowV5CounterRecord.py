import struct
import logging
class sFlowV5CounterRecord(object):
    _PACK_STR = '!II'
    MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, enterprise, counter_data_format, counter_data_length, counter_data):
        super(sFlowV5CounterRecord, self).__init__()
        self.enterprise = enterprise
        self.counter_data_format = counter_data_format
        self.counter_data_length = counter_data_length
        self.counter_data = counter_data

    @classmethod
    def parser(cls, buf, offset):
        counterdata_format, counter_data_length = struct.unpack_from(cls._PACK_STR, buf, offset)
        format_mask = 4095
        enterprise_shiftbit = 12
        counter_data_format = counterdata_format & format_mask
        enterprise = counterdata_format >> enterprise_shiftbit
        offset += cls.MIN_LEN
        if counter_data_format == 1:
            counter_data = sFlowV5GenericInterfaceCounters.parser(buf, offset)
        else:
            LOG.info('Unknown format. ' + 'sFlowV5CounterRecord.counter_data_format=%d' % counter_data_format)
            pack_str = '!%sc' % counter_data_length
            counter_data = struct.unpack_from(pack_str, buf, offset)
        msg = cls(enterprise, counter_data_format, counter_data_length, counter_data)
        return msg