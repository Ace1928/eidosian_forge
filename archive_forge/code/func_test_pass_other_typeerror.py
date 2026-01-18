import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_pass_other_typeerror():

    class Behavior:

        def __init__(self, name):
            super().__init__()
            raise TypeError('this is a typeerror unrelated to object')

    class Widget2(Behavior, EventDispatcher):
        pass

    class Widget3(EventDispatcher, Behavior):
        pass
    for cls in [Widget2, Widget3]:
        with pytest.raises(TypeError) as cm:
            cls(name='Pasta')
        assert 'this is a typeerror unrelated to object' == str(cm.value)