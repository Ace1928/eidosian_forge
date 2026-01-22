from unittest import mock
import sqlalchemy as sa
from sqlalchemy import orm
from oslo_db.sqlalchemy import test_migrations as migrate
from oslo_db.tests.sqlalchemy import base as db_test_base
class ModelThatShouldNotBeCompared(BASE):
    __tablename__ = 'testtbl2'
    id = sa.Column('id', sa.Integer, primary_key=True)
    spam = sa.Column('spam', sa.String(10), nullable=False)