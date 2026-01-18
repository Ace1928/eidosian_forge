from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def update_resource_class(self, resource_class, **attrs):
    """Update a resource class

        :param resource_class: The value can be either the ID of a resource
            class or an
            :class:`~openstack.placement.v1.resource_class.ResourceClass`,
            instance.
        :param attrs: The attributes to update on the resource class
            represented by ``resource_class``.

        :returns: The updated resource class
        :rtype: :class:`~openstack.placement.v1.resource_class.ResourceClass`
        """
    return self._update(_resource_class.ResourceClass, resource_class, **attrs)