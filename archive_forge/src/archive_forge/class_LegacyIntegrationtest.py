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
class LegacyIntegrationtest(db_test_base._DbTestCase):

    def test_legacy_integration(self):
        legacy_facade = enginefacade.get_legacy_facade()
        self.assertTrue(legacy_facade.get_engine() is enginefacade._context_manager._factory._writer_engine)
        self.assertTrue(enginefacade.get_legacy_facade() is legacy_facade)

    def test_get_sessionmaker(self):
        legacy_facade = enginefacade.get_legacy_facade()
        self.assertTrue(legacy_facade.get_sessionmaker() is enginefacade._context_manager._factory._writer_maker)

    def test_legacy_facades_from_different_context_managers(self):
        transaction_context1 = enginefacade.transaction_context()
        transaction_context2 = enginefacade.transaction_context()
        transaction_context1.configure(connection='sqlite:///?conn1')
        transaction_context2.configure(connection='sqlite:///?conn2')
        legacy1 = transaction_context1.get_legacy_facade()
        legacy2 = transaction_context2.get_legacy_facade()
        self.assertNotEqual(legacy1, legacy2)

    def test_legacy_not_started(self):
        factory = enginefacade._TransactionFactory()
        self.assertRaises(exception.CantStartEngineError, factory.get_legacy_facade)
        legacy_facade = factory.get_legacy_facade()
        self.assertRaises(exception.CantStartEngineError, legacy_facade.get_session)
        self.assertRaises(exception.CantStartEngineError, legacy_facade.get_session)
        self.assertRaises(exception.CantStartEngineError, legacy_facade.get_engine)