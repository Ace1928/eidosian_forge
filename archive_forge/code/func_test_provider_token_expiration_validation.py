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
def test_provider_token_expiration_validation(self):
    token = token_model.TokenModel()
    token.issued_at = '2013-05-21T00:02:43.941473Z'
    token.expires_at = utils.isotime(CURRENT_DATE)
    self.assertRaises(exception.TokenNotFound, PROVIDERS.token_provider_api._is_valid_token, token)
    token = token_model.TokenModel()
    token.issued_at = '2013-05-21T00:02:43.941473Z'
    token.expires_at = utils.isotime(timeutils.utcnow() + FUTURE_DELTA)
    self.assertIsNone(PROVIDERS.token_provider_api._is_valid_token(token))