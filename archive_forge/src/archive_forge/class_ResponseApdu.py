import struct
from pyu2f import errors
class ResponseApdu(object):
    """Represents a Response APDU.

  Represents a Response APU sent by the security key.  Encoding
  is specified in FIDO U2F standards.
  """
    body = None
    sw1 = None
    sw2 = None

    def __init__(self, data):
        self.dbg_full_packet = data
        if not data or len(data) < 2:
            raise errors.InvalidResponseError()
        if len(data) > 2:
            self.body = data[:-2]
        self.sw1 = data[-2]
        self.sw2 = data[-1]

    def IsSuccess(self):
        return self.sw1 == 144 and self.sw2 == 0

    def CheckSuccessOrRaise(self):
        if self.sw1 == 105 and self.sw2 == 133:
            raise errors.TUPRequiredError()
        elif self.sw1 == 106 and self.sw2 == 128:
            raise errors.InvalidKeyHandleError()
        elif self.sw1 == 105 and self.sw2 == 132:
            raise errors.InvalidKeyHandleError()
        elif not self.IsSuccess():
            raise errors.ApduError(self.sw1, self.sw2)