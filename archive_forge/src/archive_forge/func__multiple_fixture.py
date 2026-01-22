from unittest import mock
from urllib import parse
import fixtures
import sqlalchemy
from sqlalchemy import Boolean, Index, Integer, DateTime, String
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import column_property
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import sql
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql import select
from sqlalchemy.types import UserDefinedType
from oslo_db import exception
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def _multiple_fixture(self):
    callable_fn = mock.Mock()
    for targ in [callable_fn.default, callable_fn.sqlite, callable_fn.mysql, callable_fn.mysql_pymysql, callable_fn.postgresql, callable_fn.postgresql_psycopg2, callable_fn.pyodbc]:
        targ.return_value = None
    dispatcher = orig = utils.dispatch_for_dialect('*', multiple=True)(callable_fn.default)
    dispatcher = dispatcher.dispatch_for('sqlite')(callable_fn.sqlite)
    dispatcher = dispatcher.dispatch_for('mysql+pymysql')(callable_fn.mysql_pymysql)
    dispatcher = dispatcher.dispatch_for('mysql')(callable_fn.mysql)
    dispatcher = dispatcher.dispatch_for('postgresql+*')(callable_fn.postgresql)
    dispatcher = dispatcher.dispatch_for('postgresql+psycopg2')(callable_fn.postgresql_psycopg2)
    dispatcher = dispatcher.dispatch_for('*+pyodbc')(callable_fn.pyodbc)
    self.assertTrue(dispatcher is orig)
    return (dispatcher, callable_fn)