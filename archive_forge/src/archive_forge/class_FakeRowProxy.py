from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
class FakeRowProxy(RowProxy):

    def __init__(self, arg=None):
        self.row = dict(arg)

    def __getitem__(self, key):
        return self.row.__getitem__(key)

    def keys(self):
        return self.row.keys()