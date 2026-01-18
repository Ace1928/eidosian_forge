from unittest import mock
from oslo_db import exception as db_exception
from glance.cmd import manage
from glance import context
from glance.db.sqlalchemy import api as db_api
import glance.tests.utils as test_utils
def test_purge_invalid_age_in_days(self):
    age_in_days = 'abcd'
    ex = self.assertRaises(SystemExit, self.commands.purge, age_in_days)
    expected = 'Invalid int value for age_in_days: %(age_in_days)s' % {'age_in_days': age_in_days}
    self.assertEqual(expected, ex.code)