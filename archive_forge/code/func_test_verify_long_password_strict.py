import datetime
import fixtures
import uuid
import freezegun
from oslo_config import fixture as config_fixture
from oslo_log import log
from keystone.common import fernet_utils
from keystone.common import utils as common_utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.server.flask import application
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import utils
def test_verify_long_password_strict(self):
    self.config_fixture.config(strict_password_check=False)
    self.config_fixture.config(group='identity', max_password_length=5)
    max_length = CONF.identity.max_password_length
    invalid_password = 'passw0rd'
    trunc = common_utils.verify_length_and_trunc_password(invalid_password)
    self.assertEqual(invalid_password.encode('utf-8')[:max_length], trunc)