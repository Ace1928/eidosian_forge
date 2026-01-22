from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
class TestJsonify(PecanTestCase):

    def test_simple_jsonify(self):
        Person = make_person()

        @jsonify.when_type(Person)
        def jsonify_person(obj):
            return dict(name=obj.name)

        class RootController(object):

            @expose('json')
            def index(self):
                p = Person('Jonathan', 'LaCour')
                return p
        app = TestApp(Pecan(RootController()))
        r = app.get('/')
        assert r.status_int == 200
        assert loads(r.body.decode()) == {'name': 'Jonathan LaCour'}