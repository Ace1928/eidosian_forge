import unittest
from io import BytesIO, StringIO
from testtools.compat import _b
import subunit
def test_stats_smoke_everything(self):
    self.setUpUsedStream()
    self.assertEqual(5, self.result.total_tests)
    self.assertEqual(2, self.result.passed_tests)
    self.assertEqual(2, self.result.failed_tests)
    self.assertEqual(1, self.result.skipped_tests)
    self.assertEqual({'global', 'local'}, self.result.seen_tags)