from unittest import mock
from oslo_db import exception as db_exception
from glance.cmd import manage
from glance import context
from glance.db.sqlalchemy import api as db_api
import glance.tests.utils as test_utils
@mock.patch.object(db_api, 'purge_deleted_rows')
@mock.patch.object(context, 'get_admin_context')
def test_purge_max_rows(self, mock_context, mock_db_purge):
    mock_context.return_value = self.context
    value = 2 ** 31 - 1
    self.commands.purge(age_in_days=1, max_rows=value)
    mock_db_purge.assert_called_once_with(self.context, 1, value)