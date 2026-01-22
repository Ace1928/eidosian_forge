import os
from alembic import command
from alembic import config
def _make_alembic_config():
    path = os.path.join(os.path.dirname(__file__), 'alembic', 'alembic.ini')
    return config.Config(path)