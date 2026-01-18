import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_observer(self, set_name):
    from kivy.properties import Property
    a = Property(-1)
    if set_name:
        a.set_name(wid, 'a')
        a.link_eagerly(wid)
    else:
        a.link(wid, 'a')
        a.link_deps(wid, 'a')
    self.assertEqual(a.get(wid), -1)
    global observe_called
    observe_called = 0

    def observe(obj, value):
        global observe_called
        observe_called = 1
    a.bind(wid, observe)
    a.set(wid, 0)
    self.assertEqual(a.get(wid), 0)
    self.assertEqual(observe_called, 1)
    observe_called = 0
    a.set(wid, 0)
    self.assertEqual(a.get(wid), 0)
    self.assertEqual(observe_called, 0)
    a.set(wid, 1)
    self.assertEqual(a.get(wid), 1)
    self.assertEqual(observe_called, 1)