import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_parser_numeric_2(self):
    Builder = self.import_builder()
    Builder.load_string('<TLangClass>:\n\tobj: (0.5, 0.5, 0.5)')
    wid = TLangClass()
    Builder.apply(wid)
    self.assertEqual(wid.obj, (0.5, 0.5, 0.5))