import os
import platform
import unittest
import pygame
import time
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