from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
class TestJsonifySQLAlchemyGenericEncoder(PecanTestCase):

    def setUp(self):
        super(TestJsonifySQLAlchemyGenericEncoder, self).setUp()
        if not create_engine:
            self.create_fake_proxies()
        else:
            self.create_sa_proxies()

    def create_fake_proxies(self):

        class FakeSAObject(object):

            def __init__(self):
                self._sa_class_manager = object()
                self._sa_instance_state = 'awesome'
                self.id = 1
                self.first_name = 'Jonathan'
                self.last_name = 'LaCour'

        class FakeResultProxy(ResultProxy):

            def __init__(self):
                self.rowcount = -1
                self.rows = []

            def __iter__(self):
                return iter(self.rows)

            def append(self, row):
                self.rows.append(row)

        class FakeRowProxy(RowProxy):

            def __init__(self, arg=None):
                self.row = dict(arg)

            def __getitem__(self, key):
                return self.row.__getitem__(key)

            def keys(self):
                return self.row.keys()
        self.sa_object = FakeSAObject()
        self.result_proxy = FakeResultProxy()
        self.result_proxy.append(FakeRowProxy([('id', 1), ('first_name', 'Jonathan'), ('last_name', 'LaCour')]))
        self.result_proxy.append(FakeRowProxy([('id', 2), ('first_name', 'Ryan'), ('last_name', 'Petrello')]))
        self.row_proxy = FakeRowProxy([('id', 1), ('first_name', 'Jonathan'), ('last_name', 'LaCour')])

    def create_sa_proxies(self):
        mapper_registry = registry()
        metadata = schema.MetaData()
        user_table = schema.Table('user', metadata, schema.Column('id', types.Integer, primary_key=True), schema.Column('first_name', types.Unicode(25)), schema.Column('last_name', types.Unicode(25)))

        class User(object):
            pass
        mapper_registry.map_imperatively(User, user_table)
        engine = create_engine('sqlite:///:memory:')
        metadata.bind = engine
        metadata.create_all(metadata.bind)
        session = orm.sessionmaker(bind=engine)()
        session.add(User(first_name='Jonathan', last_name='LaCour'))
        session.add(User(first_name='Ryan', last_name='Petrello'))
        session.commit()
        self.sa_object = session.query(User).first()
        select = user_table.select()
        self.result_proxy = session.execute(select)
        self.row_proxy = session.execute(select).fetchone()

    def test_sa_object(self):
        result = encode(self.sa_object)
        assert loads(result) == {'id': 1, 'first_name': 'Jonathan', 'last_name': 'LaCour'}

    def test_result_proxy(self):
        result = encode(self.result_proxy)
        assert loads(result) == {'count': 2, 'rows': [{'id': 1, 'first_name': 'Jonathan', 'last_name': 'LaCour'}, {'id': 2, 'first_name': 'Ryan', 'last_name': 'Petrello'}]}

    def test_row_proxy(self):
        result = encode(self.row_proxy)
        assert loads(result) == {'id': 1, 'first_name': 'Jonathan', 'last_name': 'LaCour'}