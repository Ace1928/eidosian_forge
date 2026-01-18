import hashlib
import socket
import time
from pyu2f import errors
from pyu2f import hardware
from pyu2f import hidtransport
from pyu2f import model
Authenticates app_id with the security key.

    Executes the U2F authentication/signature flow with the security key.

    Args:
      app_id: The app_id to register the security key against.
      challenge: Server challenge passed to the security key as a bytes object.
      registered_keys: List of keys already registered for this app_id+user.

    Returns:
      SignResponse with client_data, key_handle, and signature_data.  The client
      data is an object, while the signature_data is encoded in FIDO U2F binary
      format.

    Raises:
      U2FError: There was some kind of problem with authentication (e.g.
        there was a timeout while waiting for the test of user presence.)
    