from alembic import op
from sqlalchemy import Column, Enum
from glance.cmd import manage
from glance.db import migration
from glance.db.sqlalchemy.schema import Boolean
def _add_visibility_column(bind):
    enum = Enum('private', 'public', 'shared', 'community', name='image_visibility')
    enum.create(bind=bind)
    v_col = Column('visibility', enum, nullable=True, server_default=None)
    op.add_column('images', v_col)
    op.create_index('visibility_image_idx', 'images', ['visibility'])