import base64
import json
from pyu2f import errors
class ClientData(object):
    """FIDO U2F ClientData.

  Implements the ClientData object of the FIDO U2F protocol.
  """
    TYP_AUTHENTICATION = 'navigator.id.getAssertion'
    TYP_REGISTRATION = 'navigator.id.finishEnrollment'

    def __init__(self, typ, raw_server_challenge, origin):
        if typ not in [ClientData.TYP_REGISTRATION, ClientData.TYP_AUTHENTICATION]:
            raise errors.InvalidModelError()
        self.typ = typ
        self.raw_server_challenge = raw_server_challenge
        self.origin = origin

    def GetJson(self):
        """Returns JSON version of ClientData compatible with FIDO spec."""
        server_challenge_b64 = base64.urlsafe_b64encode(self.raw_server_challenge).decode()
        server_challenge_b64 = server_challenge_b64.rstrip('=')
        return json.dumps({'typ': self.typ, 'challenge': server_challenge_b64, 'origin': self.origin}, sort_keys=True)

    def __repr__(self):
        return self.GetJson()