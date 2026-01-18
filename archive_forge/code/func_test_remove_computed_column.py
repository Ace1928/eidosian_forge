import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import exclusions
from ...testing import is_
from ...testing import is_true
from ...testing import mock
from ...testing import TestBase
def test_remove_computed_column(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('user', m1, Column('id', Integer, primary_key=True), Column('foo', Integer, sa.Computed('5')))
    Table('user', m2, Column('id', Integer, primary_key=True))
    diffs = self._fixture(m1, m2)
    eq_(diffs[0][0], 'remove_column')
    eq_(diffs[0][2], 'user')
    c = diffs[0][3]
    eq_(c.name, 'foo')
    if config.requirements.computed_reflects_normally.enabled:
        is_true(isinstance(c.computed, sa.Computed))
    else:
        is_(c.computed, None)
    if config.requirements.computed_reflects_as_server_default.enabled:
        is_true(isinstance(c.server_default, sa.DefaultClause))
        eq_(str(c.server_default.arg.text), '5')
    elif config.requirements.computed_reflects_normally.enabled:
        is_true(isinstance(c.computed, sa.Computed))
    else:
        is_(c.computed, None)