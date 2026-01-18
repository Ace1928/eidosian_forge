import random
import time
from breezy import tests, timestamp
from breezy.osutils import local_time_offset
def test_parse_patch_date(self):
    self.assertEqual((0, 0), timestamp.parse_patch_date('1970-01-01 00:00:00 +0000'))
    self.assertEqual((0, -5 * 3600), timestamp.parse_patch_date('1969-12-31 19:00:00 -0500'))
    self.assertEqual((0, +5 * 3600), timestamp.parse_patch_date('1970-01-01 05:00:00 +0500'))
    self.assertEqual((1173193459, -5 * 3600), timestamp.parse_patch_date('2007-03-06 10:04:19 -0500'))
    self.assertEqual((1173193459, +3 * 60), timestamp.parse_patch_date('2007-03-06 15:07:19 +0003'))
    self.assertEqual((1173193459, -5 * 3600), timestamp.parse_patch_date('2007-03-06 10:04:19-0500'))
    self.assertEqual((1173193459, -5 * 3600), timestamp.parse_patch_date('2007-03-06     10:04:19     -0500'))