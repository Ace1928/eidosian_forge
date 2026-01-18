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
def test_using_w_now(self):
    """using -- 'now' parameter"""
    otp = self.randotp()
    self.assertIs(otp.now, _time.time)
    self.assertAlmostEqual(otp.normalize_time(None), int(_time.time()))
    counter = [123.12]

    def now():
        counter[0] += 1
        return counter[0]
    otp = self.randotp(cls=TOTP.using(now=now))
    self.assertEqual(otp.normalize_time(None), 126)
    self.assertEqual(otp.normalize_time(None), 127)
    self.assertRaises(TypeError, TOTP.using, now=123)
    msg_re = 'now\\(\\) function must return non-negative'
    self.assertRaisesRegex(AssertionError, msg_re, TOTP.using, now=lambda: 'abc')
    self.assertRaisesRegex(AssertionError, msg_re, TOTP.using, now=lambda: -1)