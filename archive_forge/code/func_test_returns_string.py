import datetime
from oslotest import base as test_base
from oslo_utils import fixture
from oslo_utils.fixture import keystoneidsentinel as keystids
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from oslo_utils import uuidutils
def test_returns_string(self):
    self.assertIsInstance(uuids.foo, str)
    self.assertIsInstance(keystids.foo, str)