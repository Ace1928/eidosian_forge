import hashlib
from binascii import hexlify
from hmac import HMAC
from twisted.cred.credentials import CramMD5Credentials, IUsernameHashedPassword
from twisted.trial.unittest import TestCase

        L{CramMD5Credentials} implements the L{IUsernameHashedPassword}
        interface.
        