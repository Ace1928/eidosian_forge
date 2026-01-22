import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
class ConverterTestCase(unittest.TestCase):

    def test_internal_converter_2x1(self):
        correct = {'rgba': b'\x01\x02\x03\xa1\x04\x05\x06\xa2', 'abgr': b'\xa1\x03\x02\x01\xa2\x06\x05\x04', 'bgra': b'\x03\x02\x01\xa1\x06\x05\x04\xa2', 'argb': b'\xa1\x01\x02\x03\xa2\x04\x05\x06', 'rgb': b'\x01\x02\x03\x04\x05\x06', 'bgr': b'\x03\x02\x01\x06\x05\x04', 'rgb_align4': b'\x01\x02\x03\x04\x05\x06\x00\x00', 'bgr_align4': b'\x03\x02\x01\x06\x05\x04\x00\x00'}
        src = correct.get
        rgba = src('rgba')
        self.assertEqual(rgba_to(rgba, 'rgba', 2, 1, 0), src('rgba'))
        self.assertEqual(rgba_to(rgba, 'abgr', 2, 1, 0), src('abgr'))
        self.assertEqual(rgba_to(rgba, 'bgra', 2, 1, 0), src('bgra'))
        self.assertEqual(rgba_to(rgba, 'argb', 2, 1, 0), src('argb'))
        self.assertEqual(rgba_to(rgba, 'rgb', 2, 1, 0), src('rgb'))
        self.assertEqual(rgba_to(rgba, 'bgr', 2, 1, 0), src('bgr'))
        self.assertEqual(rgba_to(rgba, 'rgb', 2, 1, None), src('rgb_align4'))
        self.assertEqual(rgba_to(rgba, 'bgr', 2, 1, None), src('bgr_align4'))

    def test_internal_converter_3x1(self):
        pad6 = b'\x00' * 6
        correct = {'rgba': b'\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t\xff', 'abgr': b'\xff\x03\x02\x01\xff\x06\x05\x04\xff\t\x08\x07', 'bgra': b'\x03\x02\x01\xff\x06\x05\x04\xff\t\x08\x07\xff', 'argb': b'\xff\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t', 'rgb_align2': b'\x01\x02\x03\x04\x05\x06\x07\x08\t\x00', 'bgr_align2': b'\x03\x02\x01\x06\x05\x04\t\x08\x07\x00', 'rgb_align8': b'\x01\x02\x03\x04\x05\x06\x07\x08\t\x00' + pad6, 'bgr_align8': b'\x03\x02\x01\x06\x05\x04\t\x08\x07\x00' + pad6}
        src = correct.get
        rgba = src('rgba')
        self.assertEqual(rgba_to(rgba, 'bgra', 3, 1, 0), src('bgra'))
        self.assertEqual(rgba_to(rgba, 'argb', 3, 1, 0), src('argb'))
        self.assertEqual(rgba_to(rgba, 'abgr', 3, 1, 0), src('abgr'))
        self.assertEqual(rgba_to(rgba, 'rgb', 3, 1, 10), src('rgb_align2'))
        self.assertEqual(rgba_to(rgba, 'bgr', 3, 1, 10), src('bgr_align2'))
        self.assertEqual(rgba_to(rgba, 'rgb', 3, 1, 16), src('rgb_align8'))
        self.assertEqual(rgba_to(rgba, 'bgr', 3, 1, 16), src('bgr_align8'))

    def test_internal_converter_1x3(self):
        pad5 = b'\x00' * 5
        correct = {'rgba': b'\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t\xff', 'rgb_raw': b'\x01\x02\x03\x04\x05\x06\x07\x08\t', 'bgr_raw': b'\x03\x02\x01\x06\x05\x04\t\x08\x07', 'rgb_align2': b'\x01\x02\x03\x00\x04\x05\x06\x00\x07\x08\t\x00', 'bgr_align2': b'\x03\x02\x01\x00\x06\x05\x04\x00\t\x08\x07\x00', 'rgb_align4': b'\x01\x02\x03\x00\x04\x05\x06\x00\x07\x08\t\x00', 'bgr_align4': b'\x03\x02\x01\x00\x06\x05\x04\x00\t\x08\x07\x00', 'rgb_align8': b'\x01\x02\x03' + pad5 + b'\x04\x05\x06' + pad5 + b'\x07\x08\t' + pad5, 'bgr_align8': b'\x03\x02\x01' + pad5 + b'\x06\x05\x04' + pad5 + b'\t\x08\x07' + pad5}
        src = correct.get
        rgba = src('rgba')
        self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 4), src('rgb_align2'))
        self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 4), src('bgr_align2'))
        self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, None), src('rgb_align4'))
        self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, None), src('bgr_align4'))
        self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 0), src('rgb_raw'))
        self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 0), src('bgr_raw'))
        self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 8), src('rgb_align8'))
        self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 8), src('bgr_align8'))