import fixtures
from glance_store._drivers.swift import utils as sutils
from glance_store import exceptions
from glance_store.tests import base
def test_quotes_processor_negative(self):
    negative_values = ['\'user"', '"user\'', "'user", '"user\'', "'user", '"user', '"', "'"]
    for value in negative_values:
        self.assertRaises(ValueError, self.method, value)