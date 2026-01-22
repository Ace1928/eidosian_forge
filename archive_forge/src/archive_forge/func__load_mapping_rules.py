import copy
import hashlib
from unittest import mock
import uuid
import fixtures
import http.client
import webtest
from keystone.auth import core as auth_core
from keystone.common import authorization
from keystone.common import context as keystone_context
from keystone.common import provider_api
from keystone.common import tokenless_auth
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_backend_sql
def _load_mapping_rules(self, rules):
    self.mapping = self._mapping_ref(rules=rules)
    PROVIDERS.federation_api.create_mapping(self.mapping['id'], self.mapping)
    self.proto_x509 = self._proto_ref(mapping_id=self.mapping['id'])
    self.proto_x509['id'] = self.protocol_id
    PROVIDERS.federation_api.create_protocol(self.idp['id'], self.proto_x509['id'], self.proto_x509)