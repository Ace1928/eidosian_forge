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
def test_totp_match_w_invalid_token(self):
    """match() -- invalid TotpMatch object"""
    time = 141230981
    token = '781501'
    otp = TOTP.using(now=lambda: time + 24 * 3600)(KEY3)
    self.assertRaises(exc.InvalidTokenError, otp.match, token, time + 60)