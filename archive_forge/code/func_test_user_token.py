from unittest import mock
from keystoneauth1 import exceptions as kc_exceptions
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
def test_user_token(self):
    stack_name = 'stackuser_testtoken'
    project_id = 'aproject123'
    user_id = 'aaabbcc'
    password = 'apassword'
    rsrc = self._user_create(stack_name=stack_name, project_id=project_id, user_id=user_id, password=password)
    self.fc.stack_domain_user_token.return_value = 'atoken123'
    rsrc.password = password
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual('atoken123', rsrc._user_token())
    self.fc.stack_domain_user_token.assert_called_once_with(password=password, project_id=project_id, user_id=user_id)