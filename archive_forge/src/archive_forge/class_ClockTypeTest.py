import os
import platform
import unittest
import pygame
import time
class ClockTypeTest(unittest.TestCase):
    __tags__ = ['timing']

    def test_construction(self):
        """Ensure a Clock object can be created"""
        c = Clock()
        self.assertTrue(c, 'Clock cannot be constructed')

    def test_get_fps(self):
        """test_get_fps tests pygame.time.get_fps()"""
        c = Clock()
        self.assertEqual(c.get_fps(), 0)
        self.assertTrue(type(c.get_fps()) == float)
        delta = 0.3
        self._fps_test(c, 100, delta)
        self._fps_test(c, 60, delta)
        self._fps_test(c, 30, delta)

    def _fps_test(self, clock, fps, delta):
        """ticks fps times each second, hence get_fps() should return fps"""
        delay_per_frame = 1.0 / fps
        for f in range(fps):
            clock.tick()
            time.sleep(delay_per_frame)
        self.assertAlmostEqual(clock.get_fps(), fps, delta=fps * delta)

    def test_get_rawtime(self):
        iterations = 10
        delay = 0.1
        delay_miliseconds = delay * 10 ** 3
        framerate_limit = 5
        delta = 50
        c = Clock()
        self.assertEqual(c.get_rawtime(), 0)
        for f in range(iterations):
            time.sleep(delay)
            c.tick(framerate_limit)
            c1 = c.get_rawtime()
            self.assertAlmostEqual(delay_miliseconds, c1, delta=delta)
        for f in range(iterations):
            time.sleep(delay)
            c.tick()
            c1 = c.get_rawtime()
            c2 = c.get_time()
            self.assertAlmostEqual(c1, c2, delta=delta)

    @unittest.skipIf(platform.machine() == 's390x', 'Fails on s390x')
    @unittest.skipIf(os.environ.get('CI', None), 'CI can have variable time slices, slow.')
    def test_get_time(self):
        delay = 0.1
        delay_miliseconds = delay * 10 ** 3
        iterations = 10
        delta = 50
        c = Clock()
        self.assertEqual(c.get_time(), 0)
        for i in range(iterations):
            time.sleep(delay)
            c.tick()
            c1 = c.get_time()
            self.assertAlmostEqual(delay_miliseconds, c1, delta=delta)
        for i in range(iterations):
            t0 = time.time()
            time.sleep(delay)
            c.tick()
            t1 = time.time()
            c1 = c.get_time()
            d0 = (t1 - t0) * 10 ** 3
            self.assertAlmostEqual(d0, c1, delta=delta)

    @unittest.skipIf(platform.machine() == 's390x', 'Fails on s390x')
    @unittest.skipIf(os.environ.get('CI', None), 'CI can have variable time slices, slow.')
    def test_tick(self):
        """Tests time.Clock.tick()"""
        '\n        Loops with a set delay a few times then checks what tick reports to\n        verify its accuracy. Then calls tick with a desired frame-rate and\n        verifies it is not faster than the desired frame-rate nor is it taking\n        a dramatically long time to complete\n        '
        epsilon = 5
        epsilon2 = 0.3
        epsilon3 = 20
        testing_framerate = 60
        milliseconds = 5.0
        collection = []
        c = Clock()
        c.tick()
        for i in range(100):
            time.sleep(milliseconds / 1000)
            collection.append(c.tick())
        for outlier in [min(collection), max(collection)]:
            if outlier != milliseconds:
                collection.remove(outlier)
        average_time = float(sum(collection)) / len(collection)
        self.assertAlmostEqual(average_time, milliseconds, delta=epsilon)
        c = Clock()
        collection = []
        start = time.time()
        for i in range(testing_framerate):
            collection.append(c.tick(testing_framerate))
        for outlier in [min(collection), max(collection)]:
            if outlier != round(1000 / testing_framerate):
                collection.remove(outlier)
        end = time.time()
        self.assertAlmostEqual(end - start, 1, delta=epsilon2)
        average_tick_time = float(sum(collection)) / len(collection)
        self.assertAlmostEqual(1000 / average_tick_time, testing_framerate, delta=epsilon3)

    def test_tick_busy_loop(self):
        """Test tick_busy_loop"""
        c = Clock()
        second_length = 1000
        shortfall_tolerance = 1
        sample_fps = 40
        self.assertGreaterEqual(c.tick_busy_loop(sample_fps), second_length / sample_fps - shortfall_tolerance)
        pygame.time.wait(10)
        self.assertGreaterEqual(c.tick_busy_loop(sample_fps), second_length / sample_fps - shortfall_tolerance)
        pygame.time.wait(200)
        self.assertGreaterEqual(c.tick_busy_loop(sample_fps), second_length / sample_fps - shortfall_tolerance)
        high_fps = 500
        self.assertGreaterEqual(c.tick_busy_loop(high_fps), second_length / high_fps - shortfall_tolerance)
        low_fps = 1
        self.assertGreaterEqual(c.tick_busy_loop(low_fps), second_length / low_fps - shortfall_tolerance)
        low_non_factor_fps = 35
        frame_length_without_decimal_places = int(second_length / low_non_factor_fps)
        self.assertGreaterEqual(c.tick_busy_loop(low_non_factor_fps), frame_length_without_decimal_places - shortfall_tolerance)
        high_non_factor_fps = 750
        frame_length_without_decimal_places_2 = int(second_length / high_non_factor_fps)
        self.assertGreaterEqual(c.tick_busy_loop(high_non_factor_fps), frame_length_without_decimal_places_2 - shortfall_tolerance)
        zero_fps = 0
        self.assertEqual(c.tick_busy_loop(zero_fps), 0)
        negative_fps = -1
        self.assertEqual(c.tick_busy_loop(negative_fps), 0)
        fractional_fps = 32.75
        frame_length_without_decimal_places_3 = int(second_length / fractional_fps)
        self.assertGreaterEqual(c.tick_busy_loop(fractional_fps), frame_length_without_decimal_places_3 - shortfall_tolerance)
        bool_fps = True
        self.assertGreaterEqual(c.tick_busy_loop(bool_fps), second_length / bool_fps - shortfall_tolerance)