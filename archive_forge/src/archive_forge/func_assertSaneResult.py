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
def assertSaneResult(self, result, wallet, key, tag='1', needs_recrypt=False):
    """check encrypt_key() result has expected format"""
    self.assertEqual(set(result), set(['v', 't', 'c', 's', 'k']))
    self.assertEqual(result['v'], 1)
    self.assertEqual(result['t'], tag)
    self.assertEqual(result['c'], wallet.encrypt_cost)
    self.assertEqual(len(result['s']), to_b32_size(wallet.salt_size))
    self.assertEqual(len(result['k']), to_b32_size(len(key)))
    result_key, result_needs_recrypt = wallet.decrypt_key(result)
    self.assertEqual(result_key, key)
    self.assertEqual(result_needs_recrypt, needs_recrypt)