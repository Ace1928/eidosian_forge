import hashlib
from binascii import hexlify
from hmac import HMAC
from twisted.cred.credentials import CramMD5Credentials, IUsernameHashedPassword
from twisted.trial.unittest import TestCase
def test_wrongPassword(self) -> None:
    """
        When an invalid response is set on the L{CramMD5Credentials} (one that
        is not the hex digest of the challenge, encrypted with the user's shared
        secret) and C{checkPassword} is called with the user's correct shared
        secret, it will return L{False}.
        """
    c = CramMD5Credentials()
    chal = c.getChallenge()
    c.response = hexlify(HMAC(b'thewrongsecret', chal, digestmod=hashlib.md5).digest())
    self.assertFalse(c.checkPassword(b'secret'))