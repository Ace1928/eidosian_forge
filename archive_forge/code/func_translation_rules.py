from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def translation_rules(self, props):
    return [translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.PORT], client_plugin=self.client_plugin(), finder='find_resourceid_by_name_or_id', entity='port'), translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.TAP_SERVICE], client_plugin=self.client_plugin(), finder='find_resourceid_by_name_or_id', entity='tap_service')]