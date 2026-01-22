import logging
from pyu2f import apdu
from pyu2f import errors
def InternalSendApdu(self, apdu_to_send):
    """Send an APDU to the device.

    Sends an APDU to the device, possibly falling back to the legacy
    encoding format that is not ISO7816-4 compatible.

    Args:
      apdu_to_send: The CommandApdu object to send

    Returns:
      The ResponseApdu object constructed out of the devices reply.
    """
    response = None
    if not self.use_legacy_format:
        response = apdu.ResponseApdu(self.transport.SendMsgBytes(apdu_to_send.ToByteArray()))
        if response.sw1 == 103 and response.sw2 == 0:
            self.use_legacy_format = True
            return self.InternalSendApdu(apdu_to_send)
    else:
        response = apdu.ResponseApdu(self.transport.SendMsgBytes(apdu_to_send.ToLegacyU2FByteArray()))
    return response