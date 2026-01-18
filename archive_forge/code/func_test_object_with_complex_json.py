import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_object_with_complex_json(self):
    obj = {u'foo': [u'bar', u'baz']}

    class JSONTest:

        def __json__(self):
            return ujson.encode(obj)
    d = {u'key': JSONTest()}
    output = ujson.encode(d)
    dec = ujson.decode(output)
    self.assertEqual(dec, {u'key': obj})