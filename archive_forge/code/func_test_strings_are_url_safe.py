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
def test_strings_are_url_safe(self):
    s = provider.random_urlsafe_str()
    self.assertEqual(s, urllib.parse.quote_plus(s))