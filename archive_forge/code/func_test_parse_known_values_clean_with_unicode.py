import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
def test_parse_known_values_clean_with_unicode(self):
    """parse should give known result with known input"""
    self.maxDiff = None
    encoding = 'utf-8' if sys.version_info[0] == 2 else None
    for dic in self.knownValuesClean + self.knownValuesCleanWithUnicode:
        result = parse(build(dic, encoding=encoding), unquote=True, encoding=encoding)
        self.assertEqual(dic, result)