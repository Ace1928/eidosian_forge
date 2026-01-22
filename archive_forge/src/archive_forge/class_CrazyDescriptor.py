import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
class CrazyDescriptor(object):

    def __get__(self, obj, type_):
        if obj is None:
            return lambda x: None