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
@testing.requires.schema_reflection
@testing.requires.schema_create_delete
def test_schema_cache(self, connection):
    insp = inspect(connection)
    is_false('foo_bar' in insp.get_schema_names())
    is_false(insp.has_schema('foo_bar'))
    connection.execute(DDL('CREATE SCHEMA foo_bar'))
    try:
        is_false('foo_bar' in insp.get_schema_names())
        is_false(insp.has_schema('foo_bar'))
        insp.clear_cache()
        is_true('foo_bar' in insp.get_schema_names())
        is_true(insp.has_schema('foo_bar'))
    finally:
        connection.execute(DDL('DROP SCHEMA foo_bar'))