import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
class StopWatchTest(test_base.BaseTestCase):

    def test_leftover_no_duration(self):
        watch = timeutils.StopWatch()
        watch.start()
        self.assertRaises(RuntimeError, watch.leftover)
        self.assertRaises(RuntimeError, watch.leftover, return_none=False)
        self.assertIsNone(watch.leftover(return_none=True))

    def test_no_states(self):
        watch = timeutils.StopWatch()
        self.assertRaises(RuntimeError, watch.stop)
        self.assertRaises(RuntimeError, watch.resume)

    def test_bad_expiry(self):
        self.assertRaises(ValueError, timeutils.StopWatch, -1)

    @mock.patch('oslo_utils.timeutils.now')
    def test_backwards(self, mock_now):
        mock_now.side_effect = [0, 0.5, -1.0, -1.0]
        watch = timeutils.StopWatch(0.1)
        watch.start()
        self.assertTrue(watch.expired())
        self.assertFalse(watch.expired())
        self.assertEqual(0.0, watch.elapsed())

    @mock.patch('oslo_utils.timeutils.now')
    def test_expiry(self, mock_now):
        mock_now.side_effect = monotonic_iter(incr=0.2)
        watch = timeutils.StopWatch(0.1)
        watch.start()
        self.assertTrue(watch.expired())

    @mock.patch('oslo_utils.timeutils.now')
    def test_not_expired(self, mock_now):
        mock_now.side_effect = monotonic_iter()
        watch = timeutils.StopWatch(0.1)
        watch.start()
        self.assertFalse(watch.expired())

    def test_has_started_stopped(self):
        watch = timeutils.StopWatch()
        self.assertFalse(watch.has_started())
        self.assertFalse(watch.has_stopped())
        watch.start()
        self.assertTrue(watch.has_started())
        self.assertFalse(watch.has_stopped())
        watch.stop()
        self.assertTrue(watch.has_stopped())
        self.assertFalse(watch.has_started())

    def test_no_expiry(self):
        watch = timeutils.StopWatch(0.1)
        self.assertRaises(RuntimeError, watch.expired)

    @mock.patch('oslo_utils.timeutils.now')
    def test_elapsed(self, mock_now):
        mock_now.side_effect = monotonic_iter(incr=0.2)
        watch = timeutils.StopWatch()
        watch.start()
        matcher = matchers.GreaterThan(0.19)
        self.assertThat(watch.elapsed(), matcher)

    def test_no_elapsed(self):
        watch = timeutils.StopWatch()
        self.assertRaises(RuntimeError, watch.elapsed)

    def test_no_leftover(self):
        watch = timeutils.StopWatch()
        self.assertRaises(RuntimeError, watch.leftover)
        watch = timeutils.StopWatch(1)
        self.assertRaises(RuntimeError, watch.leftover)

    @mock.patch('oslo_utils.timeutils.now')
    def test_pause_resume(self, mock_now):
        mock_now.side_effect = monotonic_iter()
        watch = timeutils.StopWatch()
        watch.start()
        watch.stop()
        elapsed = watch.elapsed()
        self.assertAlmostEqual(elapsed, watch.elapsed())
        watch.resume()
        self.assertNotEqual(elapsed, watch.elapsed())

    @mock.patch('oslo_utils.timeutils.now')
    def test_context_manager(self, mock_now):
        mock_now.side_effect = monotonic_iter()
        with timeutils.StopWatch() as watch:
            pass
        matcher = matchers.GreaterThan(0.04)
        self.assertThat(watch.elapsed(), matcher)

    @mock.patch('oslo_utils.timeutils.now')
    def test_context_manager_splits(self, mock_now):
        mock_now.side_effect = monotonic_iter()
        with timeutils.StopWatch() as watch:
            time.sleep(0.01)
            watch.split()
        self.assertRaises(RuntimeError, watch.split)
        self.assertEqual(1, len(watch.splits))

    def test_splits_stopped(self):
        watch = timeutils.StopWatch()
        watch.start()
        watch.split()
        watch.stop()
        self.assertRaises(RuntimeError, watch.split)

    def test_splits_never_started(self):
        watch = timeutils.StopWatch()
        self.assertRaises(RuntimeError, watch.split)

    @mock.patch('oslo_utils.timeutils.now')
    def test_splits(self, mock_now):
        mock_now.side_effect = monotonic_iter()
        watch = timeutils.StopWatch()
        watch.start()
        self.assertEqual(0, len(watch.splits))
        watch.split()
        self.assertEqual(1, len(watch.splits))
        self.assertEqual(watch.splits[0].elapsed, watch.splits[0].length)
        watch.split()
        splits = watch.splits
        self.assertEqual(2, len(splits))
        self.assertNotEqual(splits[0].elapsed, splits[1].elapsed)
        self.assertEqual(splits[1].length, splits[1].elapsed - splits[0].elapsed)
        watch.stop()
        self.assertEqual(2, len(watch.splits))
        watch.start()
        self.assertEqual(0, len(watch.splits))

    @mock.patch('oslo_utils.timeutils.now')
    def test_elapsed_maximum(self, mock_now):
        mock_now.side_effect = [0, 1] + [11] * 4
        watch = timeutils.StopWatch()
        watch.start()
        self.assertEqual(1, watch.elapsed())
        self.assertEqual(11, watch.elapsed())
        self.assertEqual(1, watch.elapsed(maximum=1))
        watch.stop()
        self.assertEqual(11, watch.elapsed())
        self.assertEqual(11, watch.elapsed())
        self.assertEqual(0, watch.elapsed(maximum=-1))