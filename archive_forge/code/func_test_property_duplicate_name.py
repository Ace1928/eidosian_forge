import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_property_duplicate_name():
    from kivy.event import EventDispatcher
    from kivy.properties import ObjectProperty

    class Event(EventDispatcher):
        a = ObjectProperty(5)
    event = Event()
    counter = 0
    counter2 = 0

    def callback(*args):
        nonlocal counter
        counter += 1

    def callback2(*args):
        nonlocal counter2
        counter2 += 1
    event.fbind('a', callback)
    event.create_property('a', None)
    event.fbind('a', callback2)
    event.a = 12
    assert not counter
    assert counter2 == 1