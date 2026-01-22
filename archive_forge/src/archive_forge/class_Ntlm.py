import base64
import struct
from ntlm_auth.constants import NegotiateFlags
from ntlm_auth.exceptions import NoAuthContextError
from ntlm_auth.messages import AuthenticateMessage, ChallengeMessage, \
from ntlm_auth.session_security import SessionSecurity
class Ntlm(object):

    def __init__(self, ntlm_compatibility=3):
        self._context = NtlmContext(None, None, ntlm_compatibility=ntlm_compatibility)
        self._challenge_token = None

    @property
    def negotiate_flags(self):
        return self._context.negotiate_flags

    @negotiate_flags.setter
    def negotiate_flags(self, value):
        self._context.negotiate_flags = value

    @property
    def ntlm_compatibility(self):
        return self._context.ntlm_compatibility

    @ntlm_compatibility.setter
    def ntlm_compatibility(self, value):
        self._context.ntlm_compatibility = value

    @property
    def negotiate_message(self):
        return self._context._negotiate_message

    @negotiate_message.setter
    def negotiate_message(self, value):
        self._context._negotiate_message = value

    @property
    def challenge_message(self):
        return self._context._challenge_message

    @challenge_message.setter
    def challenge_message(self, value):
        self._context._challenge_message = value

    @property
    def authenticate_message(self):
        return self._context._authenticate_message

    @authenticate_message.setter
    def authenticate_message(self, value):
        self._context._authenticate_message = value

    @property
    def session_security(self):
        return self._context._session_security

    @session_security.setter
    def session_security(self, value):
        self._context._session_security = value

    def create_negotiate_message(self, domain_name=None, workstation=None):
        self._context.domain = domain_name
        self._context.workstation = workstation
        msg = self._context.step()
        return base64.b64encode(msg)

    def parse_challenge_message(self, msg2):
        self._challenge_token = base64.b64decode(msg2)

    def create_authenticate_message(self, user_name, password, domain_name=None, workstation=None, server_certificate_hash=None):
        self._context.username = user_name
        self._context.password = password
        self._context.domain = domain_name
        self._context.workstation = workstation
        self._context._server_certificate_hash = server_certificate_hash
        msg = self._context.step(self._challenge_token)
        return base64.b64encode(msg)