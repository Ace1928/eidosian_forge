from the command line::
from collections import abc
import functools
import inspect
import itertools
import re
import types
import unittest
import warnings
from absl.testing import absltest
class CoopTestCaseBase(other_base_class, TestCase, metaclass=CoopMetaclass):
    pass