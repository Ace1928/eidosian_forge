from alembic import op
from sqlalchemy import sql
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_image_members_table():
    deleted_member_constraint = 'image_members_image_id_member_deleted_at_key'
    op.create_table('image_members', Column('id', Integer(), nullable=False), Column('image_id', String(length=36), nullable=False), Column('member', String(length=255), nullable=False), Column('can_share', Boolean(), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=True), Column('deleted_at', DateTime(), nullable=True), Column('deleted', Boolean(), nullable=False), Column('status', String(length=20), server_default='pending', nullable=False), ForeignKeyConstraint(['image_id'], ['images.id']), PrimaryKeyConstraint('id'), UniqueConstraint('image_id', 'member', 'deleted_at', name=deleted_member_constraint), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_image_members_deleted', 'image_members', ['deleted'], unique=False)
    op.create_index('ix_image_members_image_id', 'image_members', ['image_id'], unique=False)
    op.create_index('ix_image_members_image_id_member', 'image_members', ['image_id', 'member'], unique=False)