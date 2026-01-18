import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_timespan_fields(self):
    ts = TS(1, 51945, 5000)
    self.assertAlmostEqual(1.6, self.eval('$.days', ts), places=2)
    self.assertAlmostEqual(38.43, self.eval('$.hours', ts), places=2)
    self.assertAlmostEqual(2305.75, self.eval('$.minutes', ts), places=2)
    self.assertAlmostEqual(138345, self.eval('$.seconds', ts), places=1)
    self.assertEqual(138345005, self.eval('$.milliseconds', ts))
    self.assertEqual(138345005000, self.eval('$.microseconds', ts))