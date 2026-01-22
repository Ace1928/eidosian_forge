import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
class Spam(numbers.Number):

    def __add__(inner_self, other):
        self.fail('doing attribute lookup might have side effects')