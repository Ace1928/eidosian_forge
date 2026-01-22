from testtools import matchers
from heat.common import timeutils as util
from heat.tests import common
class DurationTest(common.HeatTestCase):

    def setUp(self):
        super(DurationTest, self).setUp()
        st = util.wallclock()
        mock_clock = self.patchobject(util, 'wallclock')
        mock_clock.side_effect = [st, st + 0.5]

    def test_duration_not_expired(self):
        self.assertFalse(util.Duration(1.0).expired())

    def test_duration_expired(self):
        self.assertTrue(util.Duration(0.1).expired())