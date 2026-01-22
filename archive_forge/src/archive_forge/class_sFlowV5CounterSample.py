import struct
import logging
class sFlowV5CounterSample(object):
    _PACK_STR = '!III'

    def __init__(self, sequence_number, source_id_type, source_id_index, counters_records_num, counters_records):
        super(sFlowV5CounterSample, self).__init__()
        self.sequence_number = sequence_number
        self.source_id_type = source_id_type
        self.source_id_index = source_id_index
        self.counters_records_num = counters_records_num
        self.counters_records = counters_records

    @classmethod
    def parser(cls, buf, offset):
        sequence_number, source_id, counters_records_num = struct.unpack_from(cls._PACK_STR, buf, offset)
        index_mask = 16777215
        type_shiftbit = 24
        source_id_index = source_id & index_mask
        source_id_type = source_id >> type_shiftbit
        offset += struct.calcsize(cls._PACK_STR)
        counters_records = []
        for i in range(counters_records_num):
            counter_record = sFlowV5CounterRecord.parser(buf, offset)
            offset += sFlowV5CounterRecord.MIN_LEN
            offset += counter_record.counter_data_length
            counters_records.append(counter_record)
        msg = cls(sequence_number, source_id_type, source_id_index, counters_records_num, counters_records)
        return msg