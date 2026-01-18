import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_reference_child_update(self, set_name):
    from kivy.properties import NumericProperty, ReferenceListProperty
    x = NumericProperty(0)
    if set_name:
        x.set_name(wid, 'x')
        x.link_eagerly(wid)
    else:
        x.link(wid, 'x')
        x.link_deps(wid, 'x')
    y = NumericProperty(0)
    if set_name:
        y.set_name(wid, 'y')
        y.link_eagerly(wid)
    else:
        y.link(wid, 'y')
        y.link_deps(wid, 'y')
    pos = ReferenceListProperty(x, y)
    if set_name:
        pos.set_name(wid, 'pos')
        pos.link_eagerly(wid)
    else:
        pos.link(wid, 'pos')
        pos.link_deps(wid, 'pos')
    pos.get(wid)[0] = 10
    self.assertEqual(pos.get(wid), [10, 0])
    pos.get(wid)[:] = (20, 30)
    self.assertEqual(pos.get(wid), [20, 30])