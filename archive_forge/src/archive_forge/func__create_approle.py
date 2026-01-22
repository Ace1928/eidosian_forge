import os
import uuid
from oslo_config import cfg
from oslo_utils import uuidutils
from oslotest import base
import requests
from testtools import testcase
from castellan.common import exception
from castellan.key_manager import vault_key_manager
from castellan.tests.functional import config
from castellan.tests.functional.key_manager import test_key_manager
def _create_approle(self, vault_approle, vault_policy):
    params = {'token_ttl': '60s', 'token_max_ttl': '60s', 'policies': [vault_policy], 'bind_secret_id': 'true', 'bound_cidr_list': '127.0.0.1/32'}
    self.session.post('{}/{}'.format(self.vault_url, APPROLE_ENDPOINT.format(role_name=vault_approle)), json=params)