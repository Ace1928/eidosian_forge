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
class MockThreadingLocal(object):

    def __init__(self):
        self.__dict__['state'] = collections.defaultdict(dict)

    def __deepcopy__(self, memo):
        return self

    def __getattr__(self, key):
        ns = self.state[test_instance.ident]
        try:
            return ns[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        ns = self.state[test_instance.ident]
        ns[key] = value

    def __delattr__(self, key):
        ns = self.state[test_instance.ident]
        try:
            del ns[key]
        except KeyError:
            raise AttributeError(key)