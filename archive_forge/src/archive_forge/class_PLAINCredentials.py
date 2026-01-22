import hashlib
import hmac
from zope.interface import implementer
from twisted.cred import credentials
from twisted.mail._except import IllegalClientResponse
from twisted.mail.interfaces import IChallengeResponse, IClientAuthentication
from twisted.python.compat import nativeString
@implementer(IChallengeResponse)
class PLAINCredentials(credentials.UsernamePassword):

    def __init__(self):
        credentials.UsernamePassword.__init__(self, None, None)

    def getChallenge(self):
        return b''

    def setResponse(self, response):
        parts = response.split(b'\x00')
        if len(parts) != 3:
            raise IllegalClientResponse('Malformed Response - wrong number of parts')
        useless, self.username, self.password = parts

    def moreChallenges(self):
        return False