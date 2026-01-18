import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_spec_inheritance_for_classes(self):

    class Foo(object):

        def a(self, x):
            pass

        class Bar(object):

            def f(self, y):
                pass
    class_mock = create_autospec(Foo)
    self.assertIsNot(class_mock, class_mock())
    for this_mock in (class_mock, class_mock()):
        this_mock.a(x=5)
        this_mock.a.assert_called_with(x=5)
        this_mock.a.assert_called_with(5)
        self.assertRaises(TypeError, this_mock.a, 'foo', 'bar')
        self.assertRaises(AttributeError, getattr, this_mock, 'b')
    instance_mock = create_autospec(Foo())
    instance_mock.a(5)
    instance_mock.a.assert_called_with(5)
    instance_mock.a.assert_called_with(x=5)
    self.assertRaises(TypeError, instance_mock.a, 'foo', 'bar')
    self.assertRaises(AttributeError, getattr, instance_mock, 'b')
    self.assertRaises(TypeError, instance_mock)
    instance_mock.Bar.f(6)
    instance_mock.Bar.f.assert_called_with(6)
    instance_mock.Bar.f.assert_called_with(y=6)
    self.assertRaises(AttributeError, getattr, instance_mock.Bar, 'g')
    instance_mock.Bar().f(6)
    instance_mock.Bar().f.assert_called_with(6)
    instance_mock.Bar().f.assert_called_with(y=6)
    self.assertRaises(AttributeError, getattr, instance_mock.Bar(), 'g')