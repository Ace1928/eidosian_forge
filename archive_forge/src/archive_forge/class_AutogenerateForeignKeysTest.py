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
class AutogenerateForeignKeysTest(AutogenFixtureTest, TestBase):
    __backend__ = True
    __requires__ = ('foreign_key_constraint_reflection',)

    def test_remove_fk(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('test', String(10), primary_key=True))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)), ForeignKeyConstraint(['test2'], ['some_table.test']))
        Table('some_table', m2, Column('test', String(10), primary_key=True))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)))
        diffs = self._fixture(m1, m2)
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['test2'], 'some_table', ['test'], conditional_name='servergenerated')

    def test_add_fk(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('id', Integer, primary_key=True), Column('test', String(10)))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)))
        Table('some_table', m2, Column('id', Integer, primary_key=True), Column('test', String(10)))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)), ForeignKeyConstraint(['test2'], ['some_table.test']))
        diffs = self._fixture(m1, m2)
        self._assert_fk_diff(diffs[0], 'add_fk', 'user', ['test2'], 'some_table', ['test'])

    def test_no_change(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('id', Integer, primary_key=True), Column('test', String(10)))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', Integer), ForeignKeyConstraint(['test2'], ['some_table.id']))
        Table('some_table', m2, Column('id', Integer, primary_key=True), Column('test', String(10)))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', Integer), ForeignKeyConstraint(['test2'], ['some_table.id']))
        diffs = self._fixture(m1, m2)
        eq_(diffs, [])

    def test_no_change_composite_fk(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('other_id_1', String(10)), Column('other_id_2', String(10)), ForeignKeyConstraint(['other_id_1', 'other_id_2'], ['some_table.id_1', 'some_table.id_2']))
        Table('some_table', m2, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('other_id_1', String(10)), Column('other_id_2', String(10)), ForeignKeyConstraint(['other_id_1', 'other_id_2'], ['some_table.id_1', 'some_table.id_2']))
        diffs = self._fixture(m1, m2)
        eq_(diffs, [])

    def test_casing_convention_changed_so_put_drops_first(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('test', String(10), primary_key=True))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)), ForeignKeyConstraint(['test2'], ['some_table.test'], name='MyFK'))
        Table('some_table', m2, Column('test', String(10), primary_key=True))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)), ForeignKeyConstraint(['a1'], ['some_table.test'], name='myfk'))
        diffs = self._fixture(m1, m2)
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['test2'], 'some_table', ['test'], name='MyFK' if config.requirements.fk_names.enabled else None)
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['a1'], 'some_table', ['test'], name='myfk')

    def test_add_composite_fk_with_name(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('other_id_1', String(10)), Column('other_id_2', String(10)))
        Table('some_table', m2, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('other_id_1', String(10)), Column('other_id_2', String(10)), ForeignKeyConstraint(['other_id_1', 'other_id_2'], ['some_table.id_1', 'some_table.id_2'], name='fk_test_name'))
        diffs = self._fixture(m1, m2)
        self._assert_fk_diff(diffs[0], 'add_fk', 'user', ['other_id_1', 'other_id_2'], 'some_table', ['id_1', 'id_2'], name='fk_test_name')

    @config.requirements.no_name_normalize
    def test_remove_composite_fk(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('other_id_1', String(10)), Column('other_id_2', String(10)), ForeignKeyConstraint(['other_id_1', 'other_id_2'], ['some_table.id_1', 'some_table.id_2'], name='fk_test_name'))
        Table('some_table', m2, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('other_id_1', String(10)), Column('other_id_2', String(10)))
        diffs = self._fixture(m1, m2)
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['other_id_1', 'other_id_2'], 'some_table', ['id_1', 'id_2'], conditional_name='fk_test_name')

    def test_add_fk_colkeys(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('other_id_1', String(10)), Column('other_id_2', String(10)))
        Table('some_table', m2, Column('id_1', String(10), key='tid1', primary_key=True), Column('id_2', String(10), key='tid2', primary_key=True))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('other_id_1', String(10), key='oid1'), Column('other_id_2', String(10), key='oid2'), ForeignKeyConstraint(['oid1', 'oid2'], ['some_table.tid1', 'some_table.tid2'], name='fk_test_name'))
        diffs = self._fixture(m1, m2)
        self._assert_fk_diff(diffs[0], 'add_fk', 'user', ['other_id_1', 'other_id_2'], 'some_table', ['id_1', 'id_2'], name='fk_test_name')

    def test_no_change_colkeys(self):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('other_id_1', String(10)), Column('other_id_2', String(10)), ForeignKeyConstraint(['other_id_1', 'other_id_2'], ['some_table.id_1', 'some_table.id_2']))
        Table('some_table', m2, Column('id_1', String(10), key='tid1', primary_key=True), Column('id_2', String(10), key='tid2', primary_key=True))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('other_id_1', String(10), key='oid1'), Column('other_id_2', String(10), key='oid2'), ForeignKeyConstraint(['oid1', 'oid2'], ['some_table.tid1', 'some_table.tid2']))
        diffs = self._fixture(m1, m2)
        eq_(diffs, [])