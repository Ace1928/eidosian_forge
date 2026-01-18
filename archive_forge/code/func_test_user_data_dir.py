from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
def test_user_data_dir(self):
    a = App()
    data_dir = a.user_data_dir
    assert os.path.exists(data_dir)