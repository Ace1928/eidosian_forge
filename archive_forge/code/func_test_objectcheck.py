import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_objectcheck(self, set_name):
    from kivy.properties import ObjectProperty
    a = ObjectProperty(False)
    if set_name:
        a.set_name(wid, 'a')
        a.link_eagerly(wid)
    else:
        a.link(wid, 'a')
        a.link_deps(wid, 'a')
    self.assertEqual(a.get(wid), False)
    a.set(wid, True)
    self.assertEqual(a.get(wid), True)