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
def test_slave_connection(self):
    paths = self.create_tempfiles([('db.master', ''), ('db.slave', '')], ext='')
    master_path = 'sqlite:///' + paths[0]
    slave_path = 'sqlite:///' + paths[1]
    facade = session.EngineFacade(sql_connection=master_path, slave_connection=slave_path)
    master = facade.get_engine()
    self.assertEqual(master_path, str(master.url))
    slave = facade.get_engine(use_slave=True)
    self.assertEqual(slave_path, str(slave.url))
    master_session = facade.get_session()
    self.assertEqual(master_path, str(master_session.bind.url))
    slave_session = facade.get_session(use_slave=True)
    self.assertEqual(slave_path, str(slave_session.bind.url))