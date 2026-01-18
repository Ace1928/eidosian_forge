import numpy as np
import param
import pytest
from packaging.version import Version
from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
def test_text_string_position(self):
    text = Text('A', 1, 'A')
    Points([('A', 1)]) * text
    self.assertEqual(text.x, 'A')