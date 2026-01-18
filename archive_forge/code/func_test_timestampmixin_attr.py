from collections import abc
import datetime
from unittest import mock
from sqlalchemy import Column
from sqlalchemy import Integer, String
from sqlalchemy import event
from sqlalchemy.orm import declarative_base
from oslo_db.sqlalchemy import models
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_timestampmixin_attr(self):
    methods = ('created_at', 'updated_at')
    for method in methods:
        self.assertTrue(hasattr(models.TimestampMixin, method), 'Method %s() is not found' % method)