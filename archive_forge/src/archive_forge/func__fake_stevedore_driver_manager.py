import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def _fake_stevedore_driver_manager(*args, **kwargs):
    if kwargs['name'] == 'module_init_exception':
        raise Exception('strategy module failed to initialize.')
    else:

        def ret():
            return None
        ret.driver = lambda: None
        ret.driver.__name__ = kwargs['name']
        ret.driver.get_strategy_name = lambda: kwargs['name']
        ret.driver.init = lambda: None
    return ret