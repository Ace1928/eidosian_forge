import xml.sax
from boto import handler
from boto.emr import emrobject
from boto.resultset import ResultSet
from tests.compat import unittest
def _assert_fields(self, response, **fields):
    for field, expected in fields.items():
        actual = getattr(response, field)
        self.assertEquals(expected, actual, 'Field %s: %r != %r' % (field, expected, actual))