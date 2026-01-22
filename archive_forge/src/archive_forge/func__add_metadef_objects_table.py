from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_metadef_objects_table():
    ns_id_name_constraint = 'uq_metadef_objects_namespace_id_name'
    op.create_table('metadef_objects', Column('id', Integer(), nullable=False), Column('namespace_id', Integer(), nullable=False), Column('name', String(length=80), nullable=False), Column('description', Text(), nullable=True), Column('required', Text(), nullable=True), Column('json_schema', JSONEncodedDict(), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=True), ForeignKeyConstraint(['namespace_id'], ['metadef_namespaces.id']), PrimaryKeyConstraint('id'), UniqueConstraint('namespace_id', 'name', name=ns_id_name_constraint), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_metadef_objects_name', 'metadef_objects', ['name'], unique=False)