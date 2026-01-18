import unittest
import textwrap
from collections import defaultdict
def test_pure_python_auto_binding(self):

    class TestEventsPureAuto(TrackCallbacks.get_base_class()):
        instantiated_widgets = []
    widget = TestEventsPureAuto()
    widget.root_widget = None
    widget.base_widget = widget
    TestEventsPureAuto.check(self)