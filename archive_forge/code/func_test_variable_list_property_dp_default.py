import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_variable_list_property_dp_default(kivy_metrics):
    from kivy.event import EventDispatcher
    from kivy.properties import VariableListProperty
    kivy_metrics.density = 1

    class Number(EventDispatcher):
        a = VariableListProperty(['10dp', (20, 'dp'), 3, 4.0])
    number = Number()
    counter = 0

    def callback(name, *args):
        nonlocal counter
        counter += 1
    number.fbind('a', callback)
    assert list(number.a) == [10, 20, 3, 4]
    assert not counter
    kivy_metrics.density = 2
    assert counter == 1
    assert list(number.a) == [20, 40, 3, 4]
    kivy_metrics.density = 1
    assert counter == 2
    assert list(number.a) == [10, 20, 3, 4]