import logging
import os
from unittest import mock
import fixtures
from oslo_config import cfg
import sqlalchemy
from sqlalchemy.engine import base as base_engine
from sqlalchemy import exc
from sqlalchemy.pool import NullPool
from sqlalchemy import sql
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Integer, String
from sqlalchemy.orm import declarative_base
from oslo_db import exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def _assert_sql_mode(self, engine, sql_mode_present, sql_mode_non_present):
    with engine.connect() as conn:
        mode = conn.execute(sql.text("SHOW VARIABLES LIKE 'sql_mode'")).fetchone()[1]
    self.assertIn(sql_mode_present, mode)
    if sql_mode_non_present:
        self.assertNotIn(sql_mode_non_present, mode)