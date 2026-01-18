import fixtures
from glance_store._drivers.swift import utils as sutils
from glance_store import exceptions
from glance_store.tests import base
def test_quotes_processor(self):
    self.assertEqual('user', self.method('user'))
    self.assertEqual('user', self.method('"user"'))
    self.assertEqual('user', self.method("'user'"))
    self.assertEqual("user'", self.method("user'"))
    self.assertEqual('user"', self.method('user"'))