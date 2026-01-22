import logging
from functools import partial
from .action import ServiceAction
from .action import WaiterAction
from .base import ResourceMeta, ServiceResource
from .collection import CollectionFactory
from .model import ResourceModel
from .response import build_identifiers, ResourceHandler
from ..exceptions import ResourceLoadException
from ..docs import docstring
def _create_waiter(factory_self, resource_waiter_model, resource_name, service_context):
    """
        Creates a new wait method for each resource where both a waiter and
        resource model is defined.
        """
    waiter = WaiterAction(resource_waiter_model, waiter_resource_name=resource_waiter_model.name)

    def do_waiter(self, *args, **kwargs):
        waiter(self, *args, **kwargs)
    do_waiter.__name__ = str(resource_waiter_model.name)
    do_waiter.__doc__ = docstring.ResourceWaiterDocstring(resource_name=resource_name, event_emitter=factory_self._emitter, service_model=service_context.service_model, resource_waiter_model=resource_waiter_model, service_waiter_model=service_context.service_waiter_model, include_signature=False)
    return do_waiter