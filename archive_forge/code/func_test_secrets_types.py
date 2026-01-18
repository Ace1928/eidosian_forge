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
def test_secrets_types(self):
    """constructor -- 'secrets' param -- input types"""
    wallet = AppWallet()
    self.assertEqual(wallet._secrets, {})
    self.assertFalse(wallet.has_secrets)
    ref = {'1': b'aaa', '2': b'bbb'}
    wallet = AppWallet(ref)
    self.assertEqual(wallet._secrets, ref)
    self.assertTrue(wallet.has_secrets)
    wallet = AppWallet('\n 1: aaa\n# comment\n \n2: bbb   ')
    self.assertEqual(wallet._secrets, ref)
    wallet = AppWallet('1: aaa: bbb \n# comment\n \n2: bbb   ')
    self.assertEqual(wallet._secrets, {'1': b'aaa: bbb', '2': b'bbb'})
    wallet = AppWallet('{"1":"aaa","2":"bbb"}')
    self.assertEqual(wallet._secrets, ref)
    self.assertRaises(TypeError, AppWallet, 123)
    self.assertRaises(TypeError, AppWallet, '[123]')
    self.assertRaises(ValueError, AppWallet, {'1': 'aaa', '2': ''})