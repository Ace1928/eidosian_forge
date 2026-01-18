from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
def test_result_proxy(self):
    result = encode(self.result_proxy)
    assert loads(result) == {'count': 2, 'rows': [{'id': 1, 'first_name': 'Jonathan', 'last_name': 'LaCour'}, {'id': 2, 'first_name': 'Ryan', 'last_name': 'Petrello'}]}