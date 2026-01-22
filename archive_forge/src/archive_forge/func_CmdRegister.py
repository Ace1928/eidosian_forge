import logging
from pyu2f import apdu
from pyu2f import errors
def CmdRegister(self, challenge_param, app_param):
    """Register security key.

    Ask the security key to register with a particular origin & client.

    Args:
      challenge_param: Arbitrary 32 byte challenge string.
      app_param: Arbitrary 32 byte applciation parameter.

    Returns:
      A binary structure containing the key handle, attestation, and a
      signature over that by the attestation key.  The precise format
      is dictated by the FIDO U2F specs.

    Raises:
      TUPRequiredError: A Test of User Precense is required to proceed.
      ApduError: Something went wrong on the device.
    """
    self.logger.debug('CmdRegister')
    if len(challenge_param) != 32 or len(app_param) != 32:
        raise errors.InvalidRequestError()
    body = bytearray(challenge_param + app_param)
    response = self.InternalSendApdu(apdu.CommandApdu(0, apdu.CMD_REGISTER, 3, 0, body))
    response.CheckSuccessOrRaise()
    return response.body