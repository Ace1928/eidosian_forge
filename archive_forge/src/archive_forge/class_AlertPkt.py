import struct
from struct import calcsize
class AlertPkt(object):
    _ALERTMSG_PACK_STR = '!256s'
    _ALERTPKT_PART_PACK_STR = '!IIIII65535s'
    _ALERTPKT_SIZE = 65863

    def __init__(self, alertmsg, pkth, dlthdr, nethdr, transhdr, data, val, pkt, event):
        self.alertmsg = alertmsg
        self.pkth = pkth
        self.dlthdr = dlthdr
        self.nethdr = nethdr
        self.transhdr = transhdr
        self.data = data
        self.val = val
        self.pkt = pkt
        self.event = event

    @classmethod
    def parser(cls, buf):
        alertmsg = struct.unpack_from(cls._ALERTMSG_PACK_STR, buf)
        offset = calcsize(cls._ALERTMSG_PACK_STR)
        pkth = PcapPktHdr32.parser(buf, offset)
        offset += PcapPktHdr32._SIZE
        dlthdr, nethdr, transhdr, data, val, pkt = struct.unpack_from(cls._ALERTPKT_PART_PACK_STR, buf, offset)
        offset += calcsize(cls._ALERTPKT_PART_PACK_STR)
        event = Event.parser(buf, offset)
        msg = cls(alertmsg, pkth, dlthdr, nethdr, transhdr, data, val, pkt, event)
        return msg