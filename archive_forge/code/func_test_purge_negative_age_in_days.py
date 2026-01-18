from unittest import mock
from oslo_db import exception as db_exception
from glance.cmd import manage
from glance import context
from glance.db.sqlalchemy import api as db_api
import glance.tests.utils as test_utils
def test_purge_negative_age_in_days(self):
    ex = self.assertRaises(SystemExit, self.commands.purge, '-1')
    self.assertEqual('Must supply a non-negative value for age.', ex.code)