import functools
from oslo_log import log as logging
from heat.common import environment_format
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils as iso8601utils
from heat.engine import attributes
from heat.engine import environment
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.scaling import lbutils
from heat.scaling import rolling_update
from heat.scaling import template
def _create_template(self, num_instances, num_replace=0, template_version=('HeatTemplateFormatVersion', '2012-12-12')):
    """Create a template to represent autoscaled instances.

        Also see heat.scaling.template.member_definitions.
        """
    instance_definition = self._get_resource_definition()
    old_resources = grouputils.get_member_definitions(self, include_failed=True)
    definitions = list(template.member_definitions(old_resources, instance_definition, num_instances, num_replace, short_id.generate_id))
    child_env = environment.get_child_environment(self.stack.env, self.child_params(), item_to_remove=self.resource_info)
    tmpl = template.make_template(definitions, version=template_version, child_env=child_env)
    att_func, res_func = ('get_attr', 'get_resource')
    if att_func not in tmpl.functions or res_func not in tmpl.functions:
        att_func, res_func = ('Fn::GetAtt', 'Ref')
    get_attr = functools.partial(tmpl.functions[att_func], None, att_func)
    get_res = functools.partial(tmpl.functions[res_func], None, res_func)
    for odefn in self._nested_output_defns([k for k, d in definitions], get_attr, get_res):
        tmpl.add_output(odefn)
    return tmpl