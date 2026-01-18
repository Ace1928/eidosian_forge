import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_update_domain_unsafe_default(self):
    """Check default for unsafe names for ``POST /domains``."""
    unsafe_name = 'i am not / safe'
    ref = unit.new_domain_ref(name=unsafe_name)
    del ref['id']
    self.patch('/domains/%(domain_id)s' % {'domain_id': self.domain_id}, body={'domain': ref})