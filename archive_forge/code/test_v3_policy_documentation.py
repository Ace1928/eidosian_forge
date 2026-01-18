import json
import uuid
import http.client
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
Call ``DELETE /policies/{policy_id}``.