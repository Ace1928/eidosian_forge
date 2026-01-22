import unittest
import textwrap
from collections import defaultdict
class TestEventsPure(TrackCallbacks.get_base_class()):
    instantiated_widgets = []
    events_in_pre = [1, 2]
    events_in_applied = [1, 2]
    events_in_post = [1, 2]

    def __init__(self, **kwargs):
        self.fbind('on_kv_pre', lambda _: self.add(2, 'pre'))
        self.fbind('on_kv_applied', lambda _, x: self.add(2, 'applied'))
        self.fbind('on_kv_post', lambda _, x: self.add(2, 'post'))
        super(TestEventsPure, self).__init__(**kwargs)