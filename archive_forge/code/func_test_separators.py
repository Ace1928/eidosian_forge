from unittest import TestCase
import textwrap
import simplejson as json
from simplejson.compat import StringIO
def test_separators(self):
    lst = [1, 2, 3, 4]
    expect = '[\n1,\n2,\n3,\n4\n]'
    expect_spaces = '[\n1, \n2, \n3, \n4\n]'
    self.assertEqual(expect_spaces, json.dumps(lst, indent=0, separators=(', ', ': ')))
    self.assertEqual(expect, json.dumps(lst, indent=0, separators=(',', ': ')))
    self.assertEqual(expect, json.dumps(lst, indent=0))