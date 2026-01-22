import datetime
import math
import unittest
from itertools import product
import numpy as np
import pandas as pd
from holoviews import Dimension, Element
from holoviews.core.util import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY
class TestTreePathUtils(unittest.TestCase):

    def test_get_path_with_label(self):
        path = get_path(Element('Test', label='A'))
        self.assertEqual(path, ('Element', 'A'))

    def test_get_path_without_label(self):
        path = get_path(Element('Test'))
        self.assertEqual(path, ('Element',))

    def test_get_path_with_custom_group(self):
        path = get_path(Element('Test', group='Custom Group'))
        self.assertEqual(path, ('Custom_Group',))

    def test_get_path_with_custom_group_and_label(self):
        path = get_path(Element('Test', group='Custom Group', label='A'))
        self.assertEqual(path, ('Custom_Group', 'A'))

    def test_get_path_from_item_with_custom_group(self):
        path = get_path((('Custom',), Element('Test')))
        self.assertEqual(path, ('Custom',))

    def test_get_path_from_item_with_custom_group_and_label(self):
        path = get_path((('Custom', 'Path'), Element('Test')))
        self.assertEqual(path, ('Custom',))

    def test_get_path_from_item_with_custom_group_and_matching_label(self):
        path = get_path((('Custom', 'Path'), Element('Test', label='Path')))
        self.assertEqual(path, ('Custom', 'Path'))

    def test_make_path_unique_no_clash(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {}, True)
        self.assertEqual(new_path, path)

    def test_make_path_unique_clash_without_label(self):
        path = ('Element',)
        new_path = make_path_unique(path, {path: 1}, True)
        self.assertEqual(new_path, path + ('I',))

    def test_make_path_unique_clash_with_label(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {path: 1}, True)
        self.assertEqual(new_path, path + ('I',))

    def test_make_path_unique_no_clash_old(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {}, False)
        self.assertEqual(new_path, path)

    def test_make_path_unique_clash_without_label_old(self):
        path = ('Element',)
        new_path = make_path_unique(path, {path: 1}, False)
        self.assertEqual(new_path, path + ('I',))

    def test_make_path_unique_clash_with_label_old(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {path: 1}, False)
        self.assertEqual(new_path, path[:-1] + ('I',))