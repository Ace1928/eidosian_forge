import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
class TestLocationOrderStrategyModule(base.IsolatedUnitTest):
    """Test routines in glance.common.location_strategy.location_order"""

    def test_get_ordered_locations(self):
        original_locs = [{'url': 'loc1'}, {'url': 'loc2'}]
        ordered_locs = location_order.get_ordered_locations(original_locs)
        self.assertEqual(original_locs, ordered_locs)