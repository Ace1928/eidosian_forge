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
@combinations(('object',), ('name',))
@config.requirements.no_name_normalize
def test_remove_connection_fk(self, hook_type):
    m1 = MetaData()
    m2 = MetaData()
    ref = Table('ref', m1, Column('id', Integer, primary_key=True))
    t1 = Table('t', m1, Column('x', Integer), Column('y', Integer))
    t1.append_constraint(ForeignKeyConstraint([t1.c.x], [ref.c.id], name='fk1'))
    t1.append_constraint(ForeignKeyConstraint([t1.c.y], [ref.c.id], name='fk2'))
    ref = Table('ref', m2, Column('id', Integer, primary_key=True))
    Table('t', m2, Column('x', Integer), Column('y', Integer))
    if hook_type == 'object':

        def include_object(object_, name, type_, reflected, compare_to):
            return not (isinstance(object_, ForeignKeyConstraint) and type_ == 'foreign_key_constraint' and reflected and (name == 'fk1'))
        diffs = self._fixture(m1, m2, object_filters=include_object)
    elif hook_type == 'name':

        def include_name(name, type_, parent_names):
            if name == 'fk1':
                if type_ == 'index':
                    return True
                eq_(type_, 'foreign_key_constraint')
                eq_(parent_names, {'schema_name': None, 'table_name': 't', 'schema_qualified_table_name': 't'})
                return False
            else:
                return True
        diffs = self._fixture(m1, m2, name_filters=include_name)
    self._assert_fk_diff(diffs[0], 'remove_fk', 't', ['y'], 'ref', ['id'], conditional_name='fk2')
    eq_(len(diffs), 1)