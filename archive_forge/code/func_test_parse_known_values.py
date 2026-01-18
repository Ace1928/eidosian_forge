import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
def test_parse_known_values(self):
    """parse should give known result with known input (quoted)"""
    self.maxDiff = None
    for dic in self.knownValues:
        result = parse(build(dic))
        self.assertEqual(dic, result)