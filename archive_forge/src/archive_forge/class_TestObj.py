from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
class TestObj:

    def __init__(self) -> None:
        self.foo = 'foo value'
        self.bar = 'bar value'
        self.baz = 'baz value'