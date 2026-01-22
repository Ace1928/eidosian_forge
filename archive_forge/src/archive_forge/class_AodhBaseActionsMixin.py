from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
from heat.engine import translation
class AodhBaseActionsMixin:

    def handle_create(self):
        props = self.get_alarm_props(self.properties)
        props['name'] = self.physical_resource_name()
        alarm = self.client().alarm.create(props)
        self.resource_id_set(alarm['alarm_id'])

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            new_props = json_snippet.properties(self.properties_schema, self.context)
            self.client().alarm.update(self.resource_id, self.get_alarm_props(new_props))