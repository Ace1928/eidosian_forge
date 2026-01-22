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
class TestSanitization(ComparisonTestCase):
    """
    Tests of sanitize_identifier
    """

    def test_simple_pound_sanitized(self):
        sanitized = sanitize_identifier('£')
        self.assertEqual(sanitized, 'pound')

    def test_simple_digit_sanitized(self):
        sanitized = sanitize_identifier('0')
        self.assertEqual(sanitized, 'A_0')

    def test_simple_underscore_sanitized(self):
        sanitized = sanitize_identifier('_test')
        self.assertEqual(sanitized, 'A__test')

    def test_simple_alpha_sanitized(self):
        sanitized = sanitize_identifier('α')
        self.assertEqual(sanitized, 'α')

    def test_simple_a_pound_sanitized(self):
        sanitized = sanitize_identifier('a £')
        self.assertEqual(sanitized, 'A_pound')

    def test_capital_delta_sanitized(self):
        sanitized = sanitize_identifier('Δ')
        self.assertEqual(sanitized, 'Δ')

    def test_lowercase_delta_sanitized(self):
        sanitized = sanitize_identifier('δ')
        self.assertEqual(sanitized, 'δ')

    def test_simple_alpha_beta_sanitized(self):
        sanitized = sanitize_identifier('α β')
        self.assertEqual(sanitized, 'α_β')

    def test_simple_alpha_beta_underscore_sanitized(self):
        sanitized = sanitize_identifier('α_β')
        self.assertEqual(sanitized, 'α_β')

    def test_simple_alpha_beta_double_underscore_sanitized(self):
        sanitized = sanitize_identifier('α__β')
        self.assertEqual(sanitized, 'α__β')

    def test_simple_alpha_beta_mixed_underscore_space_sanitized(self):
        sanitized = sanitize_identifier('α__  β')
        self.assertEqual(sanitized, 'α__β')

    def test_alpha_times_two(self):
        sanitized = sanitize_identifier('α*2')
        self.assertEqual(sanitized, 'α_times_2')

    def test_arabic_five_sanitized(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g. arabic ٥ five and urdu ۵ five
        """
        try:
            sanitize_identifier('٥')
        except SyntaxError as e:
            assert str(e).startswith("String '٥' cannot be sanitized")

    def test_urdu_five_sanitized(self):
        try:
            sanitize_identifier('۵')
        except SyntaxError as e:
            assert str(e).startswith("String '۵' cannot be sanitized")

    def test_urdu_a_five_sanitized(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g. arabic ٥ five and urdu ۵ five
        """
        sanitized = sanitize_identifier('a ۵')
        self.assertEqual(sanitized, 'A_۵')

    def test_umlaut_sanitized(self):
        sanitized = sanitize_identifier('Festkörperphysik')
        self.assertEqual(sanitized, 'Festkörperphysik')

    def test_power_umlaut_sanitized(self):
        sanitized = sanitize_identifier('^Festkörperphysik')
        self.assertEqual(sanitized, 'power_Festkörperphysik')

    def test_custom_dollar_removal_py2(self):
        sanitize_identifier.eliminations.extend(['dollar'])
        sanitized = sanitize_identifier('$E$')
        self.assertEqual(sanitized, 'E')
        sanitize_identifier.eliminations.remove('dollar')