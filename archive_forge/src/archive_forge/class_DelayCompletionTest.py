from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import delay
from heat.engine import status
from heat.tests import common
from heat.tests import utils
from oslo_utils import fixture as utils_fixture
from oslo_utils import timeutils
class DelayCompletionTest(common.HeatTestCase):

    def setUp(self):
        super(DelayCompletionTest, self).setUp()
        self.time_fixture = utils_fixture.TimeFixture()
        self.useFixture(self.time_fixture)

    def test_complete_no_wait(self):
        now = timeutils.utcnow()
        self.time_fixture.advance_time_seconds(-1)
        self.assertEqual(True, delay.Delay._check_complete(now, 0))

    def test_complete(self):
        now = timeutils.utcnow()
        self.time_fixture.advance_time_seconds(5.1)
        self.assertEqual(True, delay.Delay._check_complete(now, 5.1))

    def test_already_complete(self):
        now = timeutils.utcnow()
        self.time_fixture.advance_time_seconds(5.1)
        self.assertEqual(True, delay.Delay._check_complete(now, 5))

    def test_incomplete_short_delay(self):
        now = timeutils.utcnow()
        self.time_fixture.advance_time_seconds(2)
        self.assertEqual(False, delay.Delay._check_complete(now, 5))

    def test_incomplete_moderate_delay(self):
        now = timeutils.utcnow()
        self.time_fixture.advance_time_seconds(2)
        poll_del = self.assertRaises(resource.PollDelay, delay.Delay._check_complete, now, 6)
        self.assertEqual(2, poll_del.period)

    def test_incomplete_long_delay(self):
        now = timeutils.utcnow()
        self.time_fixture.advance_time_seconds(0.1)
        poll_del = self.assertRaises(resource.PollDelay, delay.Delay._check_complete, now, 62)
        self.assertEqual(30, poll_del.period)