import os
import platform
import unittest
import pygame
import time
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