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
def test_thread_ctxmanager_switch_styles(self):

    @enginefacade.writer.connection
    def go_one(context):
        self.assertRaises(exception.ContextNotRequestedError, getattr, context, 'session')
        self.assertIsNotNone(context.connection)
        self.ident = 2
        go_two(context)
        self.ident = 1
        self.assertRaises(exception.ContextNotRequestedError, getattr, context, 'session')
        self.assertIsNotNone(context.connection)

    @enginefacade.reader
    def go_two(context):
        self.assertRaises(exception.ContextNotRequestedError, getattr, context, 'connection')
        self.assertIsNotNone(context.session)
    context = oslo_context.RequestContext()
    with self._patch_thread_ident():
        go_one(context)