import struct
import logging
class sFlowV5GenericInterfaceCounters(object):
    _PACK_STR = '!IIQIIQIIIIIIQIIIIII'

    def __init__(self, ifIndex, ifType, ifSpeed, ifDirection, ifAdminStatus, ifOperStatus, ifInOctets, ifInUcastPkts, ifInMulticastPkts, ifInBroadcastPkts, ifInDiscards, ifInErrors, ifInUnknownProtos, ifOutOctets, ifOutUcastPkts, ifOutMulticastPkts, ifOutBroadcastPkts, ifOutDiscards, ifOutErrors, ifPromiscuousMode):
        super(sFlowV5GenericInterfaceCounters, self).__init__()
        self.ifIndex = ifIndex
        self.ifType = ifType
        self.ifSpeed = ifSpeed
        self.ifDirection = ifDirection
        self.ifAdminStatus = ifAdminStatus
        self.ifOperStatus = ifOperStatus
        self.ifInOctets = ifInOctets
        self.ifInUcastPkts = ifInUcastPkts
        self.ifInMulticastPkts = ifInMulticastPkts
        self.ifInBroadcastPkts = ifInBroadcastPkts
        self.ifInDiscards = ifInDiscards
        self.ifInErrors = ifInErrors
        self.ifInUnknownProtos = ifInUnknownProtos
        self.ifOutOctets = ifOutOctets
        self.ifOutUcastPkts = ifOutUcastPkts
        self.ifOutMulticastPkts = ifOutMulticastPkts
        self.ifOutBroadcastPkts = ifOutBroadcastPkts
        self.ifOutDiscards = ifOutDiscards
        self.ifOutErrors = ifOutErrors
        self.ifPromiscuousMode = ifPromiscuousMode

    @classmethod
    def parser(cls, buf, offset):
        ifIndex, ifType, ifSpeed, ifDirection, ifStatus, ifInOctets, ifInUcastPkts, ifInMulticastPkts, ifInBroadcastPkts, ifInDiscards, ifInErrors, ifInUnknownProtos, ifOutOctets, ifOutUcastPkts, ifOutMulticastPkts, ifOutBroadcastPkts, ifOutDiscards, ifOutErrors, ifPromiscuousMode = struct.unpack_from(cls._PACK_STR, buf, offset)
        ifStatus_mask = 1
        ifAdminStatus_shiftbit = 1
        ifOperStatus = ifStatus & ifStatus_mask
        ifAdminStatus = ifStatus >> ifAdminStatus_shiftbit & ifStatus_mask
        msg = cls(ifIndex, ifType, ifSpeed, ifDirection, ifAdminStatus, ifOperStatus, ifInOctets, ifInUcastPkts, ifInMulticastPkts, ifInBroadcastPkts, ifInDiscards, ifInErrors, ifInUnknownProtos, ifOutOctets, ifOutUcastPkts, ifOutMulticastPkts, ifOutBroadcastPkts, ifOutDiscards, ifOutErrors, ifPromiscuousMode)
        return msg