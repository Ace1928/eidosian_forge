import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
class Behavior:

    def __init__(self, name):
        super().__init__()
        raise TypeError('this is a typeerror unrelated to object')