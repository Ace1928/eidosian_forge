from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
def test_start_raw_app(self):
    lang._delayed_start = None
    a = App()
    Clock.schedule_once(a.stop, 0.1)
    a.run()