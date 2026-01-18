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
def test_single_retval(self):
    dispatcher, callable_fn = self._single_fixture()
    callable_fn.mysql_pymysql.return_value = 5
    self.assertEqual(5, dispatcher('mysql+pymysql://u:p@h/t', 3))