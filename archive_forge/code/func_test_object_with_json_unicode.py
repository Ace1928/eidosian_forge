import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_object_with_json_unicode(self):
    output_text = u'this is the correct output'

    class JSONTest:

        def __json__(self):
            return u'"' + output_text + u'"'
    d = {u'key': JSONTest()}
    output = ujson.encode(d)
    dec = ujson.decode(output)
    self.assertEqual(dec, {u'key': output_text})