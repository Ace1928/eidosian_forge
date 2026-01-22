import datetime
import pickle
import unittest
from aniso8601.utcoffset import UTCOffset
class TestUTCOffset(unittest.TestCase):

    def test_pickle(self):
        testutcoffset = UTCOffset(name='UTC', minutes=0)
        utcoffsetpickle = pickle.dumps(testutcoffset)
        resultutcoffset = pickle.loads(utcoffsetpickle)
        self.assertEqual(resultutcoffset._name, testutcoffset._name)
        self.assertEqual(resultutcoffset._utcdelta, testutcoffset._utcdelta)

    def test_repr(self):
        self.assertEqual(str(UTCOffset(minutes=0)), '+0:00:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=60)), '+1:00:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=-60)), '-1:00:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=12)), '+0:12:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=-12)), '-0:12:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=83)), '+1:23:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=-83)), '-1:23:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=1440)), '+1 day, 0:00:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=-1440)), '-1 day, 0:00:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=2967)), '+2 days, 1:27:00 UTC')
        self.assertEqual(str(UTCOffset(minutes=-2967)), '-2 days, 1:27:00 UTC')

    def test_dst(self):
        tzinfoobject = UTCOffset(minutes=240)
        result = datetime.datetime.now(tzinfoobject)
        self.assertEqual(result.tzinfo.utcoffset(None), datetime.timedelta(hours=4))