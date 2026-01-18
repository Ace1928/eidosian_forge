import pytest
from kivy.tests import async_run, UnitKivyApp, GraphicUnitTest
from unittest.mock import patch
def videoplayer_app():
    from kivy.app import App
    from kivy.uix.videoplayer import VideoPlayer

    class TestApp(UnitKivyApp, App):

        def build(self):
            root = VideoPlayer()
            return root
    return TestApp()