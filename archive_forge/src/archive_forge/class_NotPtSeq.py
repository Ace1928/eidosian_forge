import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
class NotPtSeq:

    @classmethod
    def from_points(cls, points):
        list(points)

    def __iter__(self):
        yield 0