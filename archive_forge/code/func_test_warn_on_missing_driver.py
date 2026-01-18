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
def test_warn_on_missing_driver(self):
    warnings = mock.Mock()

    def warn_interpolate(msg, args):
        warnings.warning(msg % (args,))
    with mock.patch('oslo_db.sqlalchemy.engines.LOG.warning', warn_interpolate):
        engines._vet_url(utils.make_url('mysql://scott:tiger@some_host/some_db'))
        engines._vet_url(utils.make_url('mysql+mysqldb://scott:tiger@some_host/some_db'))
        engines._vet_url(utils.make_url('mysql+pymysql://scott:tiger@some_host/some_db'))
        engines._vet_url(utils.make_url('postgresql+psycopg2://scott:tiger@some_host/some_db'))
        engines._vet_url(utils.make_url('postgresql://scott:tiger@some_host/some_db'))
    self.assertEqual([mock.call.warning("URL mysql://scott:***@some_host/some_db does not contain a '+drivername' portion, and will make use of a default driver.  A full dbname+drivername:// protocol is recommended. For MySQL, it is strongly recommended that mysql+pymysql:// be specified for maximum service compatibility"), mock.call.warning("URL postgresql://scott:***@some_host/some_db does not contain a '+drivername' portion, and will make use of a default driver.  A full dbname+drivername:// protocol is recommended.")], warnings.mock_calls)