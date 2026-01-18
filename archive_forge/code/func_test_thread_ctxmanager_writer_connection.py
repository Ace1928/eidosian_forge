import collections
import contextlib
import copy
import fixtures
import pickle
import sys
from unittest import mock
import warnings
from oslo_config import cfg
from oslo_context import context as oslo_context
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines as oslo_engines
from oslo_db.sqlalchemy import orm
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db import warning
def test_thread_ctxmanager_writer_connection(self):
    context = oslo_context.RequestContext()
    with self._patch_thread_ident():
        with enginefacade.writer.connection.using(context) as conn:
            self._assert_ctx_connection(context, conn)
            self.ident = 2
            with enginefacade.reader.connection.using(context) as conn2:
                self.assertIsNot(conn2, conn)
                self._assert_ctx_connection(context, conn2)
                with enginefacade.reader.connection.using(context) as conn3:
                    self.assertIsNot(conn3, conn)
                    self.assertIs(conn3, conn2)
            self.ident = 1
            with enginefacade.reader.connection.using(context) as conn3:
                self.assertIs(conn3, conn)
                self._assert_ctx_connection(context, conn)