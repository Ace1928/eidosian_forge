import struct
import logging
class sFlowV5FlowSample(object):
    _PACK_STR = '!IIIIIIII'

    def __init__(self, sequence_number, source_id_type, source_id_index, sampling_rate, sample_pool, drops, input_if, output_if, flow_records_num, flow_records):
        super(sFlowV5FlowSample, self).__init__()
        self.sequence_number = sequence_number
        self.source_id_type = source_id_type
        self.source_id_index = source_id_index
        self.sampling_rate = sampling_rate
        self.sample_pool = sample_pool
        self.drops = drops
        self.input_if = input_if
        self.output_if = output_if
        self.flow_records_num = flow_records_num
        self.flow_records = flow_records

    @classmethod
    def parser(cls, buf, offset):
        sequence_number, source_id, sampling_rate, sample_pool, drops, input_if, output_if, flow_records_num = struct.unpack_from(cls._PACK_STR, buf, offset)
        index_mask = 16777215
        type_shiftbit = 24
        source_id_index = source_id & index_mask
        source_id_type = source_id >> type_shiftbit
        offset += struct.calcsize(cls._PACK_STR)
        flow_records = []
        for i in range(flow_records_num):
            flow_record = sFlowV5FlowRecord.parser(buf, offset)
            offset += sFlowV5FlowRecord.MIN_LEN + flow_record.flow_data_length
            flow_records.append(flow_record)
        msg = cls(sequence_number, source_id_type, source_id_index, sampling_rate, sample_pool, drops, input_if, output_if, flow_records_num, flow_records)
        return msg