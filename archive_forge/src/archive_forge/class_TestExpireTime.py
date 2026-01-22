from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import pymacaroons
import pyrfc3339
from pymacaroons import Macaroon
class TestExpireTime(TestCase):

    def test_expire_time(self):
        ExpireTest = namedtuple('ExpireTest', 'about caveats expectTime')
        tests = [ExpireTest(about='no caveats', caveats=[], expectTime=None), ExpireTest(about='single time-before caveat', caveats=[fpcaveat(checkers.time_before_caveat(t1).condition)], expectTime=t1), ExpireTest(about='multiple time-before caveat', caveats=[fpcaveat(checkers.time_before_caveat(t2).condition), fpcaveat(checkers.time_before_caveat(t1).condition)], expectTime=t1), ExpireTest(about='mixed caveats', caveats=[fpcaveat(checkers.time_before_caveat(t1).condition), fpcaveat('allow bar'), fpcaveat(checkers.time_before_caveat(t2).condition), fpcaveat('deny foo')], expectTime=t1), ExpireTest(about='mixed caveats', caveats=[fpcaveat(checkers.COND_TIME_BEFORE + ' tomorrow')], expectTime=None)]
        for test in tests:
            print('test ', test.about)
            t = checkers.expiry_time(checkers.Namespace(), test.caveats)
            self.assertEqual(t, test.expectTime)

    def test_macaroons_expire_time(self):
        ExpireTest = namedtuple('ExpireTest', 'about macaroons expectTime')
        tests = [ExpireTest(about='no macaroons', macaroons=[newMacaroon()], expectTime=None), ExpireTest(about='single macaroon without caveats', macaroons=[newMacaroon()], expectTime=None), ExpireTest(about='multiple macaroon without caveats', macaroons=[newMacaroon()], expectTime=None), ExpireTest(about='single macaroon with time-before caveat', macaroons=[newMacaroon([checkers.time_before_caveat(t1).condition])], expectTime=t1), ExpireTest(about='single macaroon with multiple time-before caveats', macaroons=[newMacaroon([checkers.time_before_caveat(t2).condition, checkers.time_before_caveat(t1).condition])], expectTime=t1), ExpireTest(about='multiple macaroons with multiple time-before caveats', macaroons=[newMacaroon([checkers.time_before_caveat(t3).condition, checkers.time_before_caveat(t1).condition]), newMacaroon([checkers.time_before_caveat(t3).condition, checkers.time_before_caveat(t1).condition])], expectTime=t1)]
        for test in tests:
            print('test ', test.about)
            t = checkers.macaroons_expiry_time(checkers.Namespace(), test.macaroons)
            self.assertEqual(t, test.expectTime)

    def test_macaroons_expire_time_skips_third_party(self):
        m1 = newMacaroon([checkers.time_before_caveat(t1).condition])
        m2 = newMacaroon()
        m2.add_third_party_caveat('https://example.com', 'a-key', '123')
        t = checkers.macaroons_expiry_time(checkers.Namespace(), [m1, m2])
        self.assertEqual(t1, t)