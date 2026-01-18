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
def test_normalize_token_instance(self, otp=None):
    """normalize_token() -- instance method"""
    if otp is None:
        otp = self.randotp(digits=7)
    self.assertEqual(otp.normalize_token(u('1234567')), '1234567')
    self.assertEqual(otp.normalize_token(b'1234567'), '1234567')
    self.assertEqual(otp.normalize_token(1234567), '1234567')
    self.assertEqual(otp.normalize_token(234567), '0234567')
    self.assertRaises(TypeError, otp.normalize_token, 1234567.0)
    self.assertRaises(TypeError, otp.normalize_token, None)
    self.assertRaises(exc.MalformedTokenError, otp.normalize_token, '123456')
    self.assertRaises(exc.MalformedTokenError, otp.normalize_token, '01234567')
    self.assertRaises(exc.MalformedTokenError, otp.normalize_token, 12345678)