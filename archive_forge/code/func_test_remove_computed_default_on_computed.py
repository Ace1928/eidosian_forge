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
@config.requirements.computed_reflects_as_server_default
def test_remove_computed_default_on_computed(self):
    """Asserts the current behavior which is that on PG and Oracle,
        the GENERATED ALWAYS AS is reflected as a server default which we can't
        tell is actually "computed", so these come out as a modification to
        the server default.

        """
    m1 = MetaData()
    m2 = MetaData()
    Table('user', m1, Column('id', Integer, primary_key=True), Column('bar', Integer), Column('foo', Integer, sa.Computed('bar + 42')))
    Table('user', m2, Column('id', Integer, primary_key=True), Column('bar', Integer), Column('foo', Integer))
    diffs = self._fixture(m1, m2)
    eq_(diffs[0][0][0], 'modify_default')
    eq_(diffs[0][0][2], 'user')
    eq_(diffs[0][0][3], 'foo')
    old = diffs[0][0][-2]
    new = diffs[0][0][-1]
    is_(new, None)
    is_true(isinstance(old, sa.DefaultClause))
    if exclusions.against(config, 'postgresql'):
        eq_(str(old.arg.text), '(bar + 42)')
    elif exclusions.against(config, 'oracle'):
        eq_(str(old.arg.text), '"BAR"+42')