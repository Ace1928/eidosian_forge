from collections import OrderedDict
from decimal import Decimal
import io
import itertools
import json
import string
import unittest
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.json import read_json, ReadOptions, ParseOptions
class TestSerialJSONRead(BaseTestJSONRead, unittest.TestCase):

    def read_json(self, *args, **kwargs):
        read_options = kwargs.setdefault('read_options', ReadOptions())
        read_options.use_threads = False
        table = read_json(*args, **kwargs)
        table.validate(full=True)
        return table