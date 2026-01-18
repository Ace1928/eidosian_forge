from unittest import mock
from urllib import parse
import fixtures
import sqlalchemy
from sqlalchemy import Boolean, Index, Integer, DateTime, String
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import column_property
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import sql
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql import select
from sqlalchemy.types import UserDefinedType
from oslo_db import exception
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_paginate_query_no_pagination_no_sort_dirs(self):
    self.query.order_by.return_value = self.query
    self.mock_asc.side_effect = ['asc_3', 'asc_2', 'asc_1']
    utils.paginate_query(self.query, self.model, 5, ['user_id', 'project_id', 'snapshot_id'])
    self.mock_asc.assert_has_calls([mock.call(self.model.user_id), mock.call(self.model.project_id), mock.call(self.model.snapshot_id)])
    self.query.order_by.assert_has_calls([mock.call('asc_3'), mock.call('asc_2'), mock.call('asc_1')])
    self.query.limit.assert_called_once_with(5)