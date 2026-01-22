import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
@attr.s
class ABAttr(object):
    a = attr.ib()
    b = attr.ib()