import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
class Issue966(Sequence):

    @staticmethod
    def cmethod(number: int, lst: List[int]=[]):
        """
                Return a list of numbers

                Example:
                ========
                C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
                """
        return lst + [number]

    @staticmethod
    def bmethod(number, lst):
        """
                Return a list of numbers

                Example:
                ========
                C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
                """
        return lst + [number]