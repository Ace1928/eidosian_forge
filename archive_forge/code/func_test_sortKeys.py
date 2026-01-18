import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_sortKeys(self):
    data = {'a': 1, 'c': 1, 'b': 1, 'e': 1, 'f': 1, 'd': 1}
    sortedKeys = ujson.dumps(data, sort_keys=True)
    self.assertEqual(sortedKeys, '{"a":1,"b":1,"c":1,"d":1,"e":1,"f":1}')