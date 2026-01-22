import operator
import re
import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock
class NormalizedNameTest(fixtures.TablesTest):
    __requires__ = ('denormalized_names',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(quoted_name('t1', quote=True), metadata, Column('id', Integer, primary_key=True))
        Table(quoted_name('t2', quote=True), metadata, Column('id', Integer, primary_key=True), Column('t1id', ForeignKey('t1.id')))

    def test_reflect_lowercase_forced_tables(self):
        m2 = MetaData()
        t2_ref = Table(quoted_name('t2', quote=True), m2, autoload_with=config.db)
        t1_ref = m2.tables['t1']
        assert t2_ref.c.t1id.references(t1_ref.c.id)
        m3 = MetaData()
        m3.reflect(config.db, only=lambda name, m: name.lower() in ('t1', 't2'))
        assert m3.tables['t2'].c.t1id.references(m3.tables['t1'].c.id)

    def test_get_table_names(self):
        tablenames = [t for t in inspect(config.db).get_table_names() if t.lower() in ('t1', 't2')]
        eq_(tablenames[0].upper(), tablenames[0].lower())
        eq_(tablenames[1].upper(), tablenames[1].lower())