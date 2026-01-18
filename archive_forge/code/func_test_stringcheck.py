import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_stringcheck(self, set_name):
    from kivy.properties import StringProperty
    a = StringProperty()
    if set_name:
        a.set_name(wid, 'a')
        a.link_eagerly(wid)
    else:
        a.link(wid, 'a')
        a.link_deps(wid, 'a')
    self.assertEqual(a.get(wid), '')
    a.set(wid, 'hello')
    self.assertEqual(a.get(wid), 'hello')
    try:
        a.set(wid, 88)
        self.fail('string accept number, fail.')
    except ValueError:
        pass