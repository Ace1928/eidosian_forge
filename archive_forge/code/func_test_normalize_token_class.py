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
def test_normalize_token_class(self):
    """normalize_token() -- class method"""
    self.test_normalize_token_instance(otp=TOTP.using(digits=7))