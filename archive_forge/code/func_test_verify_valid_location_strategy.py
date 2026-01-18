import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def test_verify_valid_location_strategy(self):
    for strategy_name in ['location_order', 'store_type']:
        self.config(location_strategy=strategy_name)
        location_strategy.verify_location_strategy()