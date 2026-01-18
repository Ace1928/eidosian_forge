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
@mock.patch('oslo_utils.timeutils.utcnow')
def test_soft_delete(self, mock_utcnow):
    dt = datetime.datetime.utcnow().replace(microsecond=0)
    mock_utcnow.return_value = dt
    m = SoftDeletedModel(id=123456, smth='test')
    self.session.add(m)
    self.session.commit()
    self.assertEqual(0, m.deleted)
    self.assertIsNone(m.deleted_at)
    m.soft_delete(self.session)
    self.assertEqual(123456, m.deleted)
    self.assertIs(dt, m.deleted_at)