from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine import attributes
from heat.engine import properties
from heat.engine.resources.openstack.neutron import net
from heat.engine.resources.openstack.neutron import neutron as nr
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _get_some_neutron_resource(self):

    class SomeNeutronResource(nr.NeutronResource):
        properties_schema = {}

        @classmethod
        def is_service_available(cls, context):
            return (True, None)
    empty_tmpl = {'heat_template_version': 'ocata'}
    tmpl = template.Template(empty_tmpl)
    stack_name = 'dummystack'
    self.dummy_stack = stack.Stack(utils.dummy_context(), stack_name, tmpl)
    self.dummy_stack.store()
    tmpl = rsrc_defn.ResourceDefinition('test_res', 'Foo')
    return SomeNeutronResource('aresource', tmpl, self.dummy_stack)