from pyu2f import hidtransport
from pyu2f.hid import base
def InternalGenerateReply(self):
    if self.init_packet.cmd == hidtransport.UsbHidTransport.U2FHID_INIT:
        nonce = self.init_packet.payload[0:8]
        self.reply = nonce + self.cid_to_allocate + bytearray(b'\x01\x00\x00\x00\x00')
    elif self.init_packet.cmd == hidtransport.UsbHidTransport.U2FHID_PING:
        self.reply = self.init_packet.payload
    elif self.init_packet.cmd == hidtransport.UsbHidTransport.U2FHID_MSG:
        self.reply = self.msg_reply
    else:
        raise UnsupportedCommandError()