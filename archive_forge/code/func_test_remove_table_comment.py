from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ...testing import eq_
from ...testing import mock
from ...testing import TestBase
def test_remove_table_comment(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('some_table', m1, Column('test', String(10), primary_key=True), comment='this is some table')
    Table('some_table', m2, Column('test', String(10), primary_key=True))
    diffs = self._fixture(m1, m2)
    eq_(diffs[0][0], 'remove_table_comment')
    eq_(diffs[0][1].comment, None)