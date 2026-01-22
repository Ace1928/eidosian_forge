from decimal import Decimal
import uuid
from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid
class ReturningTest(fixtures.TablesTest):
    run_create_tables = 'each'
    __requires__ = ('insert_returning', 'autoincrement_insert')
    __backend__ = True

    def _assert_round_trip(self, table, conn):
        row = conn.execute(table.select()).first()
        eq_(row, (conn.dialect.default_sequence_base, 'some data'))

    @classmethod
    def define_tables(cls, metadata):
        Table('autoinc_pk', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('data', String(50)))

    @requirements.fetch_rows_post_commit
    def test_explicit_returning_pk_autocommit(self, connection):
        table = self.tables.autoinc_pk
        r = connection.execute(table.insert().returning(table.c.id), dict(data='some data'))
        pk = r.first()[0]
        fetched_pk = connection.scalar(select(table.c.id))
        eq_(fetched_pk, pk)

    def test_explicit_returning_pk_no_autocommit(self, connection):
        table = self.tables.autoinc_pk
        r = connection.execute(table.insert().returning(table.c.id), dict(data='some data'))
        pk = r.first()[0]
        fetched_pk = connection.scalar(select(table.c.id))
        eq_(fetched_pk, pk)

    def test_autoincrement_on_insert_implicit_returning(self, connection):
        connection.execute(self.tables.autoinc_pk.insert(), dict(data='some data'))
        self._assert_round_trip(self.tables.autoinc_pk, connection)

    def test_last_inserted_id_implicit_returning(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert(), dict(data='some data'))
        pk = connection.scalar(select(self.tables.autoinc_pk.c.id))
        eq_(r.inserted_primary_key, (pk,))

    @requirements.insert_executemany_returning
    def test_insertmanyvalues_returning(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert().returning(self.tables.autoinc_pk.c.id), [{'data': 'd1'}, {'data': 'd2'}, {'data': 'd3'}, {'data': 'd4'}, {'data': 'd5'}])
        rall = r.all()
        pks = connection.execute(select(self.tables.autoinc_pk.c.id))
        eq_(rall, pks.all())

    @testing.combinations((Double(), 8.5514716, True), (Double(53), 8.5514716, True, testing.requires.float_or_double_precision_behaves_generically), (Float(), 8.5514, True), (Float(8), 8.5514, True, testing.requires.float_or_double_precision_behaves_generically), (Numeric(precision=15, scale=12, asdecimal=False), 8.5514716, True, testing.requires.literal_float_coercion), (Numeric(precision=15, scale=12, asdecimal=True), Decimal('8.5514716'), False), argnames='type_,value,do_rounding')
    @testing.variation('sort_by_parameter_order', [True, False])
    @testing.variation('multiple_rows', [True, False])
    def test_insert_w_floats(self, connection, metadata, sort_by_parameter_order, type_, value, do_rounding, multiple_rows):
        """test #9701.

        this tests insertmanyvalues as well as decimal / floating point
        RETURNING types

        """
        t = Table('f_t', metadata, Column('id', Integer, Identity(), primary_key=True), Column('value', type_))
        t.create(connection)
        result = connection.execute(t.insert().returning(t.c.id, t.c.value, sort_by_parameter_order=bool(sort_by_parameter_order)), [{'value': value} for i in range(10)] if multiple_rows else {'value': value})
        if multiple_rows:
            i_range = range(1, 11)
        else:
            i_range = range(1, 2)
        if do_rounding:
            eq_({(id_, round(val_, 5)) for id_, val_ in result}, {(id_, round(value, 5)) for id_ in i_range})
            eq_({round(val_, 5) for val_ in connection.scalars(select(t.c.value))}, {round(value, 5)})
        else:
            eq_(set(result), {(id_, value) for id_ in i_range})
            eq_(set(connection.scalars(select(t.c.value))), {value})

    @testing.combinations(('non_native_uuid', Uuid(native_uuid=False), uuid.uuid4()), ('non_native_uuid_str', Uuid(as_uuid=False, native_uuid=False), str(uuid.uuid4())), ('generic_native_uuid', Uuid(native_uuid=True), uuid.uuid4(), testing.requires.uuid_data_type), ('generic_native_uuid_str', Uuid(as_uuid=False, native_uuid=True), str(uuid.uuid4()), testing.requires.uuid_data_type), ('UUID', UUID(), uuid.uuid4(), testing.requires.uuid_data_type), ('LargeBinary1', LargeBinary(), b'this is binary'), ('LargeBinary2', LargeBinary(), b'7\xe7\x9f'), argnames='type_,value', id_='iaa')
    @testing.variation('sort_by_parameter_order', [True, False])
    @testing.variation('multiple_rows', [True, False])
    @testing.requires.insert_returning
    def test_imv_returning_datatypes(self, connection, metadata, sort_by_parameter_order, type_, value, multiple_rows):
        """test #9739, #9808 (similar to #9701).

        this tests insertmanyvalues in conjunction with various datatypes.

        These tests are particularly for the asyncpg driver which needs
        most types to be explicitly cast for the new IMV format

        """
        t = Table('d_t', metadata, Column('id', Integer, Identity(), primary_key=True), Column('value', type_))
        t.create(connection)
        result = connection.execute(t.insert().returning(t.c.id, t.c.value, sort_by_parameter_order=bool(sort_by_parameter_order)), [{'value': value} for i in range(10)] if multiple_rows else {'value': value})
        if multiple_rows:
            i_range = range(1, 11)
        else:
            i_range = range(1, 2)
        eq_(set(result), {(id_, value) for id_ in i_range})
        eq_(set(connection.scalars(select(t.c.value))), {value})