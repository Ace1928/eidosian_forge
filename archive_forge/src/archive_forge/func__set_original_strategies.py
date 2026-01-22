import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def _set_original_strategies(self, original_strategies):
    for name in location_strategy._available_strategies.keys():
        if name not in original_strategies:
            del location_strategy._available_strategies[name]