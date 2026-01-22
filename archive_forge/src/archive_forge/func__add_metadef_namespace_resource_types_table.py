from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_metadef_namespace_resource_types_table():
    op.create_table('metadef_namespace_resource_types', Column('resource_type_id', Integer(), nullable=False), Column('namespace_id', Integer(), nullable=False), Column('properties_target', String(length=80), nullable=True), Column('prefix', String(length=80), nullable=True), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=True), ForeignKeyConstraint(['namespace_id'], ['metadef_namespaces.id']), ForeignKeyConstraint(['resource_type_id'], ['metadef_resource_types.id']), PrimaryKeyConstraint('resource_type_id', 'namespace_id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_metadef_ns_res_types_namespace_id', 'metadef_namespace_resource_types', ['namespace_id'], unique=False)