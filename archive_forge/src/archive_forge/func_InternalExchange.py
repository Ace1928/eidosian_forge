import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def InternalExchange(self, cmd, payload_in):
    """Sends and receives a message from the device."""
    self.logger.debug('payload: ' + str(list(payload_in)))
    payload = bytearray()
    payload[:] = payload_in
    for _ in range(2):
        self.InternalSend(cmd, payload)
        ret_cmd, ret_payload = self.InternalRecv()
        if ret_cmd == UsbHidTransport.U2FHID_ERROR:
            if ret_payload == UsbHidTransport.ERR_CHANNEL_BUSY:
                time.sleep(0.5)
                continue
            raise errors.HidError('Device error: %d' % int(ret_payload[0]))
        elif ret_cmd != cmd:
            raise errors.HidError('Command mismatch!')
        return ret_payload
    raise errors.HidError('Device Busy.  Please retry')