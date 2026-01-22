from alembic import op
from sqlalchemy import Enum
from glance.cmd import manage
from glance.db import migration
def _drop_column():
    with op.batch_alter_table('images') as batch_op:
        batch_op.drop_index('ix_images_is_public')
        batch_op.drop_column('is_public')