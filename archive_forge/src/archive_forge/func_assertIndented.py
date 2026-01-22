import difflib
import inspect
import re
import unittest
from code import compile_command as compiler
from functools import partial
from bpython.curtsiesfrontend.interpreter import code_finished_will_parse
from bpython.curtsiesfrontend.preprocess import preprocess
from bpython.test.fodder import original, processed
def assertIndented(self, obj_or_name):
    if isinstance(obj_or_name, str):
        self.assertLinesIndented(obj_or_name)
    else:
        self.assertDefinitionIndented(obj_or_name)