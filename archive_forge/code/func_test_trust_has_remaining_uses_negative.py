import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_trust_has_remaining_uses_negative(self):
    self.assertRaises(exception.ValidationError, self.create_sample_trust, uuid.uuid4().hex, remaining_uses=0)
    self.assertRaises(exception.ValidationError, self.create_sample_trust, uuid.uuid4().hex, remaining_uses=-12)