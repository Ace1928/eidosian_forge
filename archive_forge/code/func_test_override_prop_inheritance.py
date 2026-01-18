import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_override_prop_inheritance():
    from kivy.event import EventDispatcher
    from kivy.properties import ObjectProperty, AliasProperty
    counter = 0

    class Parent(EventDispatcher):
        prop = ObjectProperty()

    class Child(Parent):

        def inc(self, *args):
            nonlocal counter
            counter += 1
            return counter
        prop = AliasProperty(inc)
    parent = Parent()
    child = Child()
    parent.prop = 44
    assert parent.prop == 44
    assert counter == 0
    assert child.prop == 1
    assert counter == 1
    assert parent.prop == 44
    assert isinstance(parent.property('prop'), ObjectProperty)
    assert isinstance(child.property('prop'), AliasProperty)