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
class SoftDeletedModel(BASE, models.ModelBase, models.SoftDeleteMixin):
    __tablename__ = 'test_model_soft_deletes'
    id = Column('id', Integer, primary_key=True)
    smth = Column('smth', String(255))