import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
def test_rows_cols_sizes(self):
    gl = GridLayout()
    gl.cols = 1
    gl.cols_minimum = {i: 10 for i in range(10)}
    gl.add_widget(GridLayout())
    self.render(gl)