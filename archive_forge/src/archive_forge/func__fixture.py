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
def _fixture(self, sql_mode=None, mysql_wsrep_sync_wait=None):
    kw = {}
    if sql_mode is not None:
        kw['mysql_sql_mode'] = sql_mode
    if mysql_wsrep_sync_wait is not None:
        kw['mysql_wsrep_sync_wait'] = mysql_wsrep_sync_wait
    return session.create_engine(self.engine.url, **kw)