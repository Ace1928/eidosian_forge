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
@mock.patch.object(sqlalchemy.sql, 'and_')
@mock.patch.object(sqlalchemy.sql, 'or_')
def test_paginate_query_marker_null_with_two_primary_keys(self, mock_or, mock_and):
    self.mock_asc.return_value = 'asc_1'
    self.mock_desc.side_effect = ['asc_null_2', 'desc_null_2', 'desc_1', 'desc_null_4', 'desc_4']
    self.query.order_by.return_value = self.query
    mock_or.side_effect = ['or_1', 'or_2', 'some_f']
    mock_and.side_effect = ['some_crit', 'other_crit']
    self.query.filter.return_value = self.query
    with mock.patch.object(self.model.user_id.comparator.expression, 'is_not') as mock_is_not, mock.patch.object(self.model.updated_at.comparator.expression, 'is_') as mock_is_a, mock.patch.object(self.model.user_id.comparator.expression, 'is_') as mock_is_b, mock.patch.object(self.model.project_id.comparator.expression, 'is_') as mock_is_c:
        mock_is_not.return_value = 'asc_null_1'
        mock_is_a.return_value = 'desc_null_1'
        mock_is_b.side_effect = ['asc_null_filter_1', 'asc_null_filter_2']
        mock_is_c.side_effect = ['desc_null_3', 'desc_null_filter_3']
        utils.paginate_query(self.query, self.model, 5, ['user_id', 'updated_at', 'project_id'], marker=self.marker, sort_dirs=['asc-nullslast', 'desc-nullsfirst', 'desc-nullsfirst'])
        mock_is_not.assert_called_once_with(None)
        mock_is_a.assert_called_once_with(None)
        mock_is_b.assert_has_calls([mock.call(None), mock.call(None)])
        mock_is_c.assert_has_calls([mock.call(None), mock.call(None)])
    self.mock_asc.assert_called_once_with(self.model.user_id)
    self.mock_desc.assert_has_calls([mock.call('asc_null_1'), mock.call('desc_null_1'), mock.call(self.model.updated_at), mock.call('desc_null_3'), mock.call(self.model.project_id)])
    self.query.order_by.assert_has_calls([mock.call('asc_null_2'), mock.call('asc_1'), mock.call('desc_null_2'), mock.call('desc_1'), mock.call('desc_null_4'), mock.call('desc_4')])
    mock_or.assert_has_calls([mock.call(mock.ANY, 'asc_null_filter_2'), mock.call(mock.ANY, 'desc_null_filter_3'), mock.call('some_crit', 'other_crit')])
    mock_and.assert_has_calls([mock.call('or_1'), mock.call(mock.ANY, 'or_2')])
    self.query.filter.assert_called_once_with('some_f')
    self.query.limit.assert_called_once_with(5)