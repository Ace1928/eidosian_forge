from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
def _add_artifact_properties_table():
    op.create_table('artifact_properties', Column('id', String(length=36), nullable=False), Column('artifact_id', String(length=36), nullable=False), Column('name', String(length=255), nullable=False), Column('string_value', String(length=255), nullable=True), Column('int_value', Integer(), nullable=True), Column('numeric_value', Numeric(), nullable=True), Column('bool_value', Boolean(), nullable=True), Column('text_value', Text(), nullable=True), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=False), Column('position', Integer(), nullable=True), ForeignKeyConstraint(['artifact_id'], ['artifacts.id']), PrimaryKeyConstraint('id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_artifact_properties_artifact_id', 'artifact_properties', ['artifact_id'], unique=False)
    op.create_index('ix_artifact_properties_name', 'artifact_properties', ['name'], unique=False)