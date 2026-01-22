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
class AssertDataSource(collections.namedtuple('AssertDataSource', ['writer', 'reader', 'async_reader'])):

    def element_for_writer(self, const):
        if const is enginefacade._WRITER:
            return self.writer
        elif const is enginefacade._READER:
            return self.reader
        elif const is enginefacade._ASYNC_READER:
            return self.async_reader
        else:
            assert False, 'Unknown constant: %s' % const