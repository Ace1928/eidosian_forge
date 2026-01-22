import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def _list_users_in_group_by_password_expires_at(self, time, operator=None, expected_status=http.client.OK):
    """Call `list_users_in_group` with `password_expires_at` filter.

        GET /groups/{group_id}/users?password_expires_at=
        {operator}:{timestamp}&{operator}:{timestamp}

        """
    url = '/groups/' + self.group_id + '/users?password_expires_at='
    if operator:
        url += operator + ':'
    url += str(time)
    return url