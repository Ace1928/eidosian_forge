import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import sys
import time as _time
from passlib import exc
from passlib.utils.compat import unicode, u
from passlib.tests.utils import TestCase, time_call
from passlib import totp as totp_module
from passlib.totp import TOTP, AppWallet, AES_SUPPORT
def test_to_uri(self):
    """to_uri()"""
    otp = TOTP(KEY4, alg='sha1', digits=6, period=30)
    self.assertEqual(otp.to_uri('alice@google.com', 'Example Org'), 'otpauth://totp/Example%20Org:alice@google.com?secret=JBSWY3DPEHPK3PXP&issuer=Example%20Org')
    self.assertRaises(ValueError, otp.to_uri, None, 'Example Org')
    self.assertEqual(otp.to_uri('alice@google.com'), 'otpauth://totp/alice@google.com?secret=JBSWY3DPEHPK3PXP')
    otp.label = 'alice@google.com'
    self.assertEqual(otp.to_uri(), 'otpauth://totp/alice@google.com?secret=JBSWY3DPEHPK3PXP')
    otp.issuer = 'Example Org'
    self.assertEqual(otp.to_uri(), 'otpauth://totp/Example%20Org:alice@google.com?secret=JBSWY3DPEHPK3PXP&issuer=Example%20Org')
    self.assertRaises(ValueError, otp.to_uri, 'label:with:semicolons')
    self.assertRaises(ValueError, otp.to_uri, 'alice@google.com', 'issuer:with:semicolons')
    self.assertEqual(TOTP(KEY4, alg='sha256').to_uri('alice@google.com'), 'otpauth://totp/alice@google.com?secret=JBSWY3DPEHPK3PXP&algorithm=SHA256')
    self.assertEqual(TOTP(KEY4, digits=8).to_uri('alice@google.com'), 'otpauth://totp/alice@google.com?secret=JBSWY3DPEHPK3PXP&digits=8')
    self.assertEqual(TOTP(KEY4, period=63).to_uri('alice@google.com'), 'otpauth://totp/alice@google.com?secret=JBSWY3DPEHPK3PXP&period=63')