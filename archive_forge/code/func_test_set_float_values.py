import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
@unittest.skipIf(IS_PYPY, 'fails on pypy (but only for: bottom, right, centerx, centery)')
def test_set_float_values(self):
    zero = 0
    pos = 124
    neg = -432
    data_rows = [(zero, 0.1, zero, _random_int()), (zero, 0.4, zero, _random_int()), (zero, 0.5, zero + 1, _random_int()), (zero, 1.1, zero + 1, _random_int()), (zero, 1.5, zero + 2, _random_int()), (zero, -0.1, zero, _random_int()), (zero, -0.4, zero, _random_int()), (zero, -0.5, zero - 1, _random_int()), (zero, -0.6, zero - 1, _random_int()), (zero, -1.6, zero - 2, _random_int()), (zero, 1, zero + 1, _random_int()), (zero, 4, zero + 4, _random_int()), (zero, -1, zero - 1, _random_int()), (zero, -4, zero - 4, _random_int()), (pos, 0.1, pos, _random_int()), (pos, 0.4, pos, _random_int()), (pos, 0.5, pos + 1, _random_int()), (pos, 1.1, pos + 1, _random_int()), (pos, 1.5, pos + 2, _random_int()), (pos, -0.1, pos, _random_int()), (pos, -0.4, pos, _random_int()), (pos, -0.5, pos, _random_int()), (pos, -0.6, pos - 1, _random_int()), (pos, -1.6, pos - 2, _random_int()), (pos, 1, pos + 1, _random_int()), (pos, 4, pos + 4, _random_int()), (pos, -1, pos - 1, _random_int()), (pos, -4, pos - 4, _random_int()), (neg, 0.1, neg, _random_int()), (neg, 0.4, neg, _random_int()), (neg, 0.5, neg, _random_int()), (neg, 1.1, neg + 1, _random_int()), (neg, 1.5, neg + 1, _random_int()), (neg, -0.1, neg, _random_int()), (neg, -0.4, neg, _random_int()), (neg, -0.5, neg - 1, _random_int()), (neg, -0.6, neg - 1, _random_int()), (neg, -1.6, neg - 2, _random_int()), (neg, 1, neg + 1, _random_int()), (neg, 4, neg + 4, _random_int()), (neg, -1, neg - 1, _random_int()), (neg, -4, neg - 4, _random_int())]
    single_value_attribute_names = ['x', 'y', 'w', 'h', 'width', 'height', 'top', 'left', 'bottom', 'right', 'centerx', 'centery']
    tuple_value_attribute_names = ['topleft', 'topright', 'bottomleft', 'bottomright', 'midtop', 'midleft', 'midbottom', 'midright', 'size', 'center']
    for row in data_rows:
        initial, inc, expected, other = row
        new_value = initial + inc
        for attribute_name in single_value_attribute_names:
            with self.subTest(row=row, name=f'r.{attribute_name}'):
                actual = Rect(_random_int(), _random_int(), _random_int(), _random_int())
                setattr(actual, attribute_name, new_value)
                self.assertEqual(expected, getattr(actual, attribute_name))
        for attribute_name in tuple_value_attribute_names:
            with self.subTest(row=row, name=f'r.{attribute_name}[0]'):
                actual = Rect(_random_int(), _random_int(), _random_int(), _random_int())
                setattr(actual, attribute_name, (new_value, other))
                self.assertEqual((expected, other), getattr(actual, attribute_name))
            with self.subTest(row=row, name=f'r.{attribute_name}[1]'):
                actual = Rect(_random_int(), _random_int(), _random_int(), _random_int())
                setattr(actual, attribute_name, (other, new_value))
                self.assertEqual((other, expected), getattr(actual, attribute_name))