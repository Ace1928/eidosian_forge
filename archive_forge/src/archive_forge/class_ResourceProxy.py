import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
class ResourceProxy(status.ResourceStatus):
    """A lightweight API for essential data about a resource.

    This is the interface through which template functions will access data
    about particular resources in the stack definition, such as the resource
    definition and current values of reference IDs and attributes.

    Resource proxies for some or all resources in the stack will potentially be
    loaded for every check resource operation, so it is essential that this API
    is implemented efficiently, using only the data received over RPC and
    without reference to the resource data stored in the database.

    This API can be considered stable by third-party Template or Function
    plugins, and no part of it should be changed or removed without an
    appropriate deprecation process.
    """
    __slots__ = ('name', '_definition', '_resource_data')

    def __init__(self, name, definition, resource_data):
        self.name = name
        self._definition = definition
        self._resource_data = resource_data

    @property
    def t(self):
        """The resource definition."""
        return self._definition

    def _res_data(self):
        assert self._resource_data is not None, 'Resource data not available'
        return self._resource_data

    @property
    def attributes_schema(self):
        """A set of the valid top-level attribute names.

        This is provided for backwards-compatibility for functions that require
        a container with all of the valid attribute names in order to validate
        the template. Other operations on it are invalid because we don't
        actually have access to the attributes schema here; hence we return a
        set instead of a dict.
        """
        return set(self._res_data().attribute_names())

    @property
    def external_id(self):
        """The external ID of the resource."""
        return self._definition.external_id()

    @property
    def state(self):
        """The current state (action, status) of the resource."""
        return (self.action, self.status)

    @property
    def action(self):
        """The current action of the resource."""
        if self._resource_data is None:
            return self.INIT
        return self._resource_data.action

    @property
    def status(self):
        """The current status of the resource."""
        if self._resource_data is None:
            return self.COMPLETE
        return self._resource_data.status

    def FnGetRefId(self):
        """For the intrinsic function get_resource."""
        if self._resource_data is None:
            return self.name
        return self._resource_data.reference_id()

    def FnGetAtt(self, attr, *path):
        """For the intrinsic function get_attr."""
        if path:
            attr = (attr,) + path
        try:
            return self._res_data().attribute(attr)
        except KeyError:
            raise exception.InvalidTemplateAttribute(resource=self.name, key=attr)

    def FnGetAtts(self):
        """For the intrinsic function get_attr when getting all attributes.

        :returns: a dict of all of the resource's attribute values, excluding
                  the "show" attribute.
        """
        all_attrs = self._res_data().attributes()
        return dict(((k, v) for k, v in all_attrs.items() if k != attributes.SHOW_ATTR))