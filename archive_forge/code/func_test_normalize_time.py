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
def test_normalize_time(self):
    """normalize_time()"""
    TotpFactory = TOTP.using()
    otp = self.randotp(TotpFactory)
    for _ in range(10):
        time = self.randtime()
        tint = int(time)
        self.assertEqual(otp.normalize_time(time), tint)
        self.assertEqual(otp.normalize_time(tint + 0.5), tint)
        self.assertEqual(otp.normalize_time(tint), tint)
        dt = datetime.datetime.utcfromtimestamp(time)
        self.assertEqual(otp.normalize_time(dt), tint)
        orig = TotpFactory.now
        try:
            TotpFactory.now = staticmethod(lambda: time)
            self.assertEqual(otp.normalize_time(None), tint)
        finally:
            TotpFactory.now = orig
    self.assertRaises(TypeError, otp.normalize_time, '1234')