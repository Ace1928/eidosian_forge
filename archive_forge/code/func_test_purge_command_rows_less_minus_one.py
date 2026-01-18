from unittest import mock
from oslo_db import exception as db_exception
from glance.cmd import manage
from glance import context
from glance.db.sqlalchemy import api as db_api
import glance.tests.utils as test_utils
def test_purge_command_rows_less_minus_one(self):
    exit = self.assertRaises(SystemExit, self.commands.purge, 1, -2)
    self.assertEqual('Minimal rows limit is -1.', exit.code)