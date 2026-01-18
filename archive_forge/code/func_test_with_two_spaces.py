import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_with_two_spaces(self):
    Builder = self.import_builder()
    Builder.load_string(dedent("\n        <TLangClass>:\n          on_press:\n            print('hello world')\n            print('this is working !')\n            self.a = 1\n        "))
    wid = TLangClass()
    Builder.apply(wid)
    wid.a = 0
    self.assertTrue('on_press' in wid.binded_func)
    wid.binded_func['on_press']()
    self.assertEqual(wid.a, 1)