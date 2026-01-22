import struct
import logging
class sFlowV5Sample(object):
    _PACK_STR = '!II'
    MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, enterprise, sample_format, sample_length, sample):
        super(sFlowV5Sample, self).__init__()
        self.enterprise = enterprise
        self.sample_format = sample_format
        self.sample_length = sample_length
        self.sample = sample

    @classmethod
    def parser(cls, buf, offset):
        sampledata_format, sample_length = struct.unpack_from(cls._PACK_STR, buf, offset)
        format_mask = 4095
        enterprise_shiftbit = 12
        sample_format = sampledata_format & format_mask
        enterprise = sampledata_format >> enterprise_shiftbit
        offset += cls.MIN_LEN
        if sample_format == 1:
            sample = sFlowV5FlowSample.parser(buf, offset)
        elif sample_format == 2:
            sample = sFlowV5CounterSample.parser(buf, offset)
        else:
            LOG.info('Unknown format. sFlowV5Sample.sample_format=%d', sample_format)
            pack_str = '!%sc' % sample_length
            sample = struct.unpack_from(pack_str, buf, offset)
        msg = cls(enterprise, sample_format, sample_length, sample)
        return msg