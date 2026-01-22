import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def _get_prop(self):
    self.getter_called += 1
    return self.base_value * 2