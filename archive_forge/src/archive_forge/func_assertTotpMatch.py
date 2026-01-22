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
def assertTotpMatch(self, match, time, skipped=0, period=30, window=30, msg=''):
    from passlib.totp import TotpMatch
    self.assertIsInstance(match, TotpMatch)
    self.assertIsInstance(match.totp, TOTP)
    self.assertEqual(match.totp.period, period)
    self.assertEqual(match.time, time, msg=msg + ' matched time:')
    expected = time // period
    counter = expected + skipped
    self.assertEqual(match.counter, counter, msg=msg + ' matched counter:')
    self.assertEqual(match.expected_counter, expected, msg=msg + ' expected counter:')
    self.assertEqual(match.skipped, skipped, msg=msg + ' skipped:')
    self.assertEqual(match.cache_seconds, period + window)
    expire_time = (counter + 1) * period
    self.assertEqual(match.expire_time, expire_time)
    self.assertEqual(match.cache_time, expire_time + window)
    self.assertEqual(len(match), 2)
    self.assertEqual(match, (counter, time))
    self.assertRaises(IndexError, match.__getitem__, -3)
    self.assertEqual(match[0], counter)
    self.assertEqual(match[1], time)
    self.assertRaises(IndexError, match.__getitem__, 2)
    self.assertTrue(match)