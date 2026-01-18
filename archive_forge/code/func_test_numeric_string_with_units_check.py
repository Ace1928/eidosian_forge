import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_numeric_string_with_units_check(self, set_name):
    from kivy.properties import NumericProperty
    from kivy.metrics import Metrics
    a = NumericProperty()
    if set_name:
        a.set_name(wid, 'a')
        a.link_eagerly(wid)
    else:
        a.link(wid, 'a')
        a.link_deps(wid, 'a')
    self.assertEqual(a.get(wid), 0)
    a.set(wid, '55dp')
    density = Metrics.density
    self.assertEqual(a.get(wid), 55 * density)
    self.assertEqual(a.get_format(wid), 'dp')
    a.set(wid, u'55dp')
    self.assertEqual(a.get(wid), 55 * density)
    self.assertEqual(a.get_format(wid), 'dp')
    a.set(wid, '99in')
    self.assertEqual(a.get(wid), 9504.0 * density)
    self.assertEqual(a.get_format(wid), 'in')
    a.set(wid, u'99in')
    self.assertEqual(a.get(wid), 9504.0 * density)
    self.assertEqual(a.get_format(wid), 'in')