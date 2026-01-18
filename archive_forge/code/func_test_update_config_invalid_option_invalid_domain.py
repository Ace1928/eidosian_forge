import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_config_invalid_option_invalid_domain(self):
    """Call ``PATCH /domains/{domain_id}/config/{group}/{invalid}``.

        While updating Identity API-based domain config with an invalid option
        and an invalid domain id provided, the request shall be rejected
        with a response, 404 domain not found.
        """
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    invalid_option = uuid.uuid4().hex
    new_config = {'ldap': {invalid_option: uuid.uuid4().hex}}
    invalid_domain_id = uuid.uuid4().hex
    self.patch('/domains/%(domain_id)s/config/ldap/%(invalid_option)s' % {'domain_id': invalid_domain_id, 'invalid_option': invalid_option}, body={'config': new_config}, expected_status=exception.DomainNotFound.code)