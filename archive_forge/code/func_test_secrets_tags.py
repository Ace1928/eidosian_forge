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
def test_secrets_tags(self):
    """constructor -- 'secrets' param -- tag/value normalization"""
    ref = {'1': b'aaa', '02': b'bbb', 'C': b'ccc'}
    wallet = AppWallet(ref)
    self.assertEqual(wallet._secrets, ref)
    wallet = AppWallet({u('1'): b'aaa', u('02'): b'bbb', u('C'): b'ccc'})
    self.assertEqual(wallet._secrets, ref)
    wallet = AppWallet({1: b'aaa', '02': b'bbb', 'C': b'ccc'})
    self.assertEqual(wallet._secrets, ref)
    self.assertRaises(TypeError, AppWallet, {(1,): 'aaa'})
    wallet = AppWallet({'1-2_3.4': b'aaa'})
    self.assertRaises(ValueError, AppWallet, {'-abc': 'aaa'})
    self.assertRaises(ValueError, AppWallet, {'ab*$': 'aaa'})
    wallet = AppWallet({'1': u('aaa'), '02': 'bbb', 'C': b'ccc'})
    self.assertEqual(wallet._secrets, ref)
    self.assertRaises(TypeError, AppWallet, {'1': 123})
    self.assertRaises(TypeError, AppWallet, {'1': None})
    self.assertRaises(TypeError, AppWallet, {'1': []})