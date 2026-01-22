import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
class CustomAlias(EventDispatcher):

    def _get_prop(self):
        self.getter_called += 1
        return self.base_value * 2

    def _set_prop(self, value):
        self.setter_called += 1
        self.base_value = value / 2
        return True
    prop = AliasProperty(_get_prop, _set_prop, cache=True, force_dispatch=True, watch_before_use=watch_before_use)

    def __init__(self, **kwargs):
        super(CustomAlias, self).__init__(**kwargs)
        self.base_value = 1
        self.getter_called = 0
        self.setter_called = 0
        self.callback_called = 0