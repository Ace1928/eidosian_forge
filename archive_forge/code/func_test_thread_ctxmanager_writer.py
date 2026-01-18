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
def test_thread_ctxmanager_writer(self):
    context = oslo_context.RequestContext()
    with self._patch_thread_ident():
        with enginefacade.writer.using(context) as session:
            self._assert_ctx_session(context, session)
            self.ident = 2
            with enginefacade.reader.using(context) as sess2:
                self.assertIsNot(sess2, session)
                self._assert_ctx_session(context, sess2)
            self.ident = 1
            with enginefacade.reader.using(context) as sess3:
                self.assertIs(sess3, session)
                self._assert_ctx_session(context, session)