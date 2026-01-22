import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
def _dialect_registries(dialect):
    if dialect.name in _registry:
        yield _registry[dialect.name]
    if '*' in _registry:
        yield _registry['*']