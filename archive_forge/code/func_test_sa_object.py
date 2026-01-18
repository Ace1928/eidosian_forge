from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
def test_sa_object(self):
    result = encode(self.sa_object)
    assert loads(result) == {'id': 1, 'first_name': 'Jonathan', 'last_name': 'LaCour'}