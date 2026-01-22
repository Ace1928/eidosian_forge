import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
class StackDefinition(object):
    """Class representing the definition of a Stack, but not its current state.

    This is the interface through which template functions will access data
    about the stack definition, including the template and current values of
    resource reference IDs and attributes.

    This API can be considered stable by third-party Template or Function
    plugins, and no part of it should be changed or removed without an
    appropriate deprecation process.
    """

    def __init__(self, context, template, stack_identifier, resource_data, parent_info=None):
        self._context = context
        self._template = template
        self._resource_data = {} if resource_data is None else resource_data
        self._parent_info = parent_info
        self._zones = None
        self.parameters = template.parameters(stack_identifier, template.env.params, template.env.param_defaults)
        self._resource_defns = None
        self._resources = {}
        self._output_defns = None

    def clone_with_new_template(self, new_template, stack_identifier, clear_resource_data=False):
        """Create a new StackDefinition with a different template."""
        res_data = {} if clear_resource_data else dict(self._resource_data)
        return type(self)(self._context, new_template, stack_identifier, res_data, self._parent_info)

    @property
    def t(self):
        """The stack's template."""
        return self._template

    @property
    def env(self):
        """The stack's environment."""
        return self._template.env

    def _load_rsrc_defns(self):
        self._resource_defns = self._template.resource_definitions(self)

    def resource_definition(self, resource_name):
        """Return the definition of the given resource."""
        if self._resource_defns is None:
            self._load_rsrc_defns()
        return self._resource_defns[resource_name]

    def enabled_rsrc_names(self):
        """Return the set of names of all enabled resources in the template."""
        if self._resource_defns is None:
            self._load_rsrc_defns()
        return set(self._resource_defns)

    def _load_output_defns(self):
        self._output_defns = self._template.outputs(self)

    def output_definition(self, output_name):
        """Return the definition of the given output."""
        if self._output_defns is None:
            self._load_output_defns()
        return self._output_defns[output_name]

    def enabled_output_names(self):
        """Return the set of names of all enabled outputs in the template."""
        if self._output_defns is None:
            self._load_output_defns()
        return set(self._output_defns)

    def all_rsrc_names(self):
        """Return the set of names of all resources in the template.

        This includes resources that are disabled due to false conditionals.
        """
        if hasattr(self._template, 'RESOURCES'):
            return set(self._template.get(self._template.RESOURCES, self._resource_defns or []))
        else:
            return self.enabled_rsrc_names()

    def all_resource_types(self):
        """Return the set of types of all resources in the template."""
        if self._resource_defns is None:
            self._load_rsrc_defns()
        return set((self._resource_defns[res].resource_type for res in self._resource_defns))

    def get_availability_zones(self):
        """Return the list of Nova availability zones."""
        if self._zones is None:
            nova = self._context.clients.client('nova')
            zones = nova.availability_zones.list(detailed=False)
            self._zones = [zone.zoneName for zone in zones]
        return self._zones

    def __contains__(self, resource_name):
        """Return True if the given resource name is present and enabled."""
        if self._resource_defns is not None:
            return resource_name in self._resource_defns
        else:
            return resource_name in self._template[self._template.RESOURCES]

    def __getitem__(self, resource_name):
        """Return a proxy for the given resource."""
        if resource_name not in self._resources:
            res_proxy = ResourceProxy(resource_name, self.resource_definition(resource_name), self._resource_data.get(resource_name))
            self._resources[resource_name] = res_proxy
        return self._resources[resource_name]

    @property
    def parent_resource(self):
        """Return a proxy for the parent resource.

        Returns None if the stack is not a provider stack for a
        TemplateResource.
        """
        return self._parent_info