from itertools import product
from kivy.tests import GraphicUnitTest
from kivy.logger import LoggerHistory
def test_window_opacity_clamping_negative(self):
    if self.check_opacity_support():
        self.Window.opacity = -1.5
        self.assertEqual(self.Window.opacity, 0.0)