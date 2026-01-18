import hashlib
from binascii import hexlify
from hmac import HMAC
from twisted.cred.credentials import CramMD5Credentials, IUsernameHashedPassword
from twisted.trial.unittest import TestCase
def test_setResponse(self) -> None:
    """
        When C{setResponse} is called with a string that is the username and
        the hashed challenge separated with a space, they will be set on the
        L{CramMD5Credentials}.
        """
    c = CramMD5Credentials()
    chal = c.getChallenge()
    c.setResponse(b' '.join((b'squirrel', hexlify(HMAC(b'supersecret', chal, digestmod=hashlib.md5).digest()))))
    self.assertTrue(c.checkPassword(b'supersecret'))
    self.assertEqual(c.username, b'squirrel')