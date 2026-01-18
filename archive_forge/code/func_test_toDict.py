import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_toDict(self):
    d = {'key': 31337}

    class DictTest:

        def toDict(self):
            return d

        def __json__(self):
            return '"json defined"'
    o = DictTest()
    output = ujson.encode(o)
    dec = ujson.decode(output)
    self.assertEqual(dec, d)