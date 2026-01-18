import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
@pytest.fixture
def widget_verify_thread(request):
    from kivy.uix.widget import Widget
    from kivy.config import Config
    original = Config.get('graphics', 'verify_gl_main_thread')
    Config.set('graphics', 'verify_gl_main_thread', request.param)
    widget = Widget()
    yield (widget, request.param)
    Config.set('graphics', 'verify_gl_main_thread', original)