import abc
import contextlib
import os
import random
import tempfile
import testtools
import sqlalchemy as sa
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_sqlalchemy
from taskflow import test
from taskflow.tests.unit.persistence import base
@testtools.skipIf(not _postgres_exists(), 'postgres is not available')
def test_postgres_persistence_entry_point(self):
    uri = _get_connect_string('postgres', USER, PASSWD, database=DATABASE)
    conf = {'connection': uri}
    with contextlib.closing(backends.fetch(conf)) as be:
        self.assertIsInstance(be, impl_sqlalchemy.SQLAlchemyBackend)