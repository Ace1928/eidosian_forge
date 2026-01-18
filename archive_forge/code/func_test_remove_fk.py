from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ...testing import combinations
from ...testing import config
from ...testing import eq_
from ...testing import mock
from ...testing import TestBase
def test_remove_fk(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('some_table', m1, Column('test', String(10), primary_key=True))
    Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)), ForeignKeyConstraint(['test2'], ['some_table.test']))
    Table('some_table', m2, Column('test', String(10), primary_key=True))
    Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)))
    diffs = self._fixture(m1, m2)
    self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['test2'], 'some_table', ['test'], conditional_name='servergenerated')