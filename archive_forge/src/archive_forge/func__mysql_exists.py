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
def _mysql_exists():
    engine = None
    try:
        db_uri = _get_connect_string('mysql', USER, PASSWD)
        engine = sa.create_engine(db_uri)
        with contextlib.closing(engine.connect()):
            return True
    except Exception:
        pass
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass
    return False