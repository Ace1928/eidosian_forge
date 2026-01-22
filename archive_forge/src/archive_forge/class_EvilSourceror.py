from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class EvilSourceror:
    a: EvilSourceror
    b: EvilSourceror
    c: object

    def __init__(self, x: object) -> None:
        self.a = self
        self.a.b = self
        self.a.b.c = x