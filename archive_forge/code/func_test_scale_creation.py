import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_scale_creation(self):
    from kivy.graphics import Scale
    self.check_transform_works(Scale)