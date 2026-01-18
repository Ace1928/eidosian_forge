import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_numeric_property_dp(kivy_metrics):
    from kivy.event import EventDispatcher
    from kivy.properties import NumericProperty
    kivy_metrics.density = 1

    class Number(EventDispatcher):
        with_dp = NumericProperty(5)
        no_dp = NumericProperty(10)
        default_dp = NumericProperty('10dp')
    number = Number()
    counter = {'with_dp': 0, 'no_dp': 0, 'default_dp': 0}

    def callback(name, *args):
        counter[name] += 1
    number.fbind('with_dp', callback, 'with_dp')
    number.fbind('no_dp', callback, 'no_dp')
    number.fbind('default_dp', callback, 'default_dp')
    assert not counter['with_dp']
    assert not counter['no_dp']
    assert not counter['default_dp']
    assert number.with_dp == 5
    assert number.no_dp == 10
    assert number.default_dp == 10
    number.with_dp = 10
    assert counter['with_dp'] == 1
    assert number.with_dp == 10
    kivy_metrics.density = 2
    assert counter['with_dp'] == 1
    assert not counter['no_dp']
    assert counter['default_dp'] == 1
    assert number.with_dp == 10
    assert number.no_dp == 10
    assert number.default_dp == 20
    number.with_dp = '20dp'
    number.no_dp = 20
    assert counter['with_dp'] == 2
    assert counter['no_dp'] == 1
    assert counter['default_dp'] == 1
    assert number.with_dp == 40
    assert number.no_dp == 20
    assert number.default_dp == 20
    kivy_metrics.density = 1
    assert counter['with_dp'] == 3
    assert counter['no_dp'] == 1
    assert counter['default_dp'] == 2
    assert number.with_dp == 20
    assert number.no_dp == 20
    assert number.default_dp == 10