import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_object_with_json_attribute_error(self):

    class JSONTest:

        def __json__(self):
            raise AttributeError
    d = {u'key': JSONTest()}
    self.assertRaises(AttributeError, ujson.encode, d)