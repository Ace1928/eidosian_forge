from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_property_config_loaded_in_order(self):
    """
        Verify the order of loaded config sections matches that from the
        configuration file
        """
    self.rules_checker = property_utils.PropertyRules(self.policy)
    self.assertEqual(CONFIG_SECTIONS, property_utils.CONFIG.sections())