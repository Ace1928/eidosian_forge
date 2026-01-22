import struct
class NetFlowV5Flow(object):
    _PACK_STR = '!IIIHHIIIIHHxBBBHHBB2x'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, srcaddr, dstaddr, nexthop, input_, output, dpkts, doctets, first, last, srcport, dstport, tcp_flags, prot, tos, src_as, dst_as, src_mask, dst_mask):
        self.srcaddr = srcaddr
        self.dstaddr = dstaddr
        self.nexthop = nexthop
        self.input = input_
        self.output = output
        self.dpkts = dpkts
        self.doctets = doctets
        self.first = first
        self.last = last
        self.srcport = srcport
        self.dstport = dstport
        self.tcp_flags = tcp_flags
        self.prot = prot
        self.tos = tos
        self.src_as = src_as
        self.dst_as = dst_as
        self.src_mask = src_mask
        self.dst_mask = dst_mask

    @classmethod
    def parser(cls, buf, offset):
        srcaddr, dstaddr, nexthop, input_, output, dpkts, doctets, first, last, srcport, dstport, tcp_flags, prot, tos, src_as, dst_as, src_mask, dst_mask = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(srcaddr, dstaddr, nexthop, input_, output, dpkts, doctets, first, last, srcport, dstport, tcp_flags, prot, tos, src_as, dst_as, src_mask, dst_mask)
        return msg