import hashlib
import socket
import time
from pyu2f import errors
from pyu2f import hardware
from pyu2f import hidtransport
from pyu2f import model
def Register(self, app_id, challenge, registered_keys):
    """Registers app_id with the security key.

    Executes the U2F registration flow with the security key.

    Args:
      app_id: The app_id to register the security key against.
      challenge: Server challenge passed to the security key.
      registered_keys: List of keys already registered for this app_id+user.

    Returns:
      RegisterResponse with key_handle and attestation information in it (
        encoded in FIDO U2F binary format within registration_data field).

    Raises:
      U2FError: There was some kind of problem with registration (e.g.
        the device was already registered or there was a timeout waiting
        for the test of user presence).
    """
    client_data = model.ClientData(model.ClientData.TYP_REGISTRATION, challenge, self.origin)
    challenge_param = self.InternalSHA256(client_data.GetJson())
    app_param = self.InternalSHA256(app_id)
    for key in registered_keys:
        try:
            if key.version != u'U2F_V2':
                continue
            resp = self.security_key.CmdAuthenticate(challenge_param, app_param, key.key_handle, True)
            raise errors.HardwareError('Should Never Happen')
        except errors.TUPRequiredError:
            raise errors.U2FError(errors.U2FError.DEVICE_INELIGIBLE)
        except errors.InvalidKeyHandleError as e:
            pass
        except errors.HardwareError as e:
            raise errors.U2FError(errors.U2FError.BAD_REQUEST, e)
    for _ in range(30):
        try:
            resp = self.security_key.CmdRegister(challenge_param, app_param)
            return model.RegisterResponse(resp, client_data)
        except errors.TUPRequiredError as e:
            self.security_key.CmdWink()
            time.sleep(0.5)
        except errors.HardwareError as e:
            raise errors.U2FError(errors.U2FError.BAD_REQUEST, e)
    raise errors.U2FError(errors.U2FError.TIMEOUT)