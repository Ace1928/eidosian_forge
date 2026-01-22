from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
class MockWithCmp(mock.MagicMock):
    order = 0

    def __init__(self, *args, **kwargs):
        super(MockWithCmp, self).__init__(*args, **kwargs)
        self.__lt__ = lambda self, other: self.order < other.order