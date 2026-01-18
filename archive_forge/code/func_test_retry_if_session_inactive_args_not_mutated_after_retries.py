from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
def test_retry_if_session_inactive_args_not_mutated_after_retries(self):
    context = mock.Mock()
    context.session.is_active = False
    list_arg = [1, 2, 3, 4]
    dict_arg = {1: 'a', 2: 'b'}
    l, d = self._context_function(context, list_arg, dict_arg, 5, db_exc.DBDeadlock())
    self.assertEqual(5, len(l))
    self.assertEqual(3, len(d))