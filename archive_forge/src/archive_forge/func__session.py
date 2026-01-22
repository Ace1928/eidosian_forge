import contextlib
from unittest import mock
from osprofiler import sqlalchemy
from osprofiler.tests import test
@contextlib.contextmanager
def _session():
    session = mock.MagicMock()
    session.bind = mock.MagicMock()
    session.bind.traced = None
    yield session