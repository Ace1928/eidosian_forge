import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_numericIntFrcExp(self):
    input = '1.337E40'
    output = ujson.decode(input)
    self.assertEqual(output, json.loads(input))