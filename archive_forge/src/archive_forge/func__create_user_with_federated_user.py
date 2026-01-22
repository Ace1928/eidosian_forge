import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def _create_user_with_federated_user(self, user, fed_dict):
    with sql.session_for_write() as session:
        federated_ref = model.FederatedUser.from_dict(fed_dict)
        user_ref = model.User.from_dict(user)
        user_ref.created_at = datetime.datetime.utcnow()
        user_ref.federated_users.append(federated_ref)
        session.add(user_ref)
        return identity_base.filter_user(user_ref.to_dict())