import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
class WithClassMethod:

    @classmethod
    def classfunc(cls):
        pass