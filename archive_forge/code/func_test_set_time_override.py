import datetime
from oslotest import base as test_base
from oslo_utils import fixture
from oslo_utils.fixture import keystoneidsentinel as keystids
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from oslo_utils import uuidutils
def test_set_time_override(self):
    new_time = datetime.datetime(2015, 1, 2, 3, 4, 6, 7)
    self.useFixture(fixture.TimeFixture(new_time))
    self.assertEqual(new_time, timeutils.utcnow())
    self.assertEqual(new_time, timeutils.utcnow())