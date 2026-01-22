import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
class TrickyDict(dict):

    def __getitem__(self, index):
        self.fail("doing key lookup isn't safe")