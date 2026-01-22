from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
def _add_node_reference_table():
    op.create_table('node_reference', Column('node_reference_id', BigInteger().with_variant(Integer, 'sqlite'), nullable=False, autoincrement=True), Column('node_reference_url', String(length=255), nullable=False), PrimaryKeyConstraint('node_reference_id'), UniqueConstraint('node_reference_url', name='uq_node_reference_node_reference_url'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)