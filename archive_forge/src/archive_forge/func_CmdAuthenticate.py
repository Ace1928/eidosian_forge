import logging
from pyu2f import apdu
from pyu2f import errors
def CmdAuthenticate(self, challenge_param, app_param, key_handle, check_only=False):
    """Attempt to obtain an authentication signature.

    Ask the security key to sign a challenge for a particular key handle
    in order to authenticate the user.

    Args:
      challenge_param: SHA-256 hash of client_data object as a bytes
          object.
      app_param: SHA-256 hash of the app id as a bytes object.
      key_handle: The key handle to use to issue the signature as a bytes
          object.
      check_only: If true, only check if key_handle is valid.

    Returns:
      A binary structure containing the key handle, attestation, and a
      signature over that by the attestation key.  The precise format
      is dictated by the FIDO U2F specs.

    Raises:
      TUPRequiredError: If check_only is False, a Test of User Precense
          is required to proceed.  If check_only is True, this means
          the key_handle is valid.
      InvalidKeyHandleError: The key_handle is not valid for this device.
      ApduError: Something else went wrong on the device.
    """
    self.logger.debug('CmdAuthenticate')
    if len(challenge_param) != 32 or len(app_param) != 32:
        raise errors.InvalidRequestError()
    control = 7 if check_only else 3
    body = bytearray(challenge_param + app_param + bytearray([len(key_handle)]) + key_handle)
    response = self.InternalSendApdu(apdu.CommandApdu(0, apdu.CMD_AUTH, control, 0, body))
    response.CheckSuccessOrRaise()
    return response.body