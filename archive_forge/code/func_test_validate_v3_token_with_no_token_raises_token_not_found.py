import datetime
from oslo_utils import timeutils
import urllib
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import token_model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone import token
from keystone.token import provider
def test_validate_v3_token_with_no_token_raises_token_not_found(self):
    self.assertRaises(exception.TokenNotFound, PROVIDERS.token_provider_api.validate_token, None)