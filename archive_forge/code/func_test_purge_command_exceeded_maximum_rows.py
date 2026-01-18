from unittest import mock
from oslo_db import exception as db_exception
from glance.cmd import manage
from glance import context
from glance.db.sqlalchemy import api as db_api
import glance.tests.utils as test_utils
def test_purge_command_exceeded_maximum_rows(self):
    value = 2 ** 31
    ex = self.assertRaises(SystemExit, self.commands.purge, age_in_days=1, max_rows=value)
    expected = "'max_rows' value out of range, must not exceed 2147483647."
    self.assertEqual(expected, ex.code)