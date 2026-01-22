import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
class DictWidgetFalse(Label):
    button = DictProperty({'button': None}, rebind=False)