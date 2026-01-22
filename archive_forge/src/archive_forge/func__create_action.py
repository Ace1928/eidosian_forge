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
def _create_action(factory_self, action_model, resource_name, service_context, is_load=False):
    """
        Creates a new method which makes a request to the underlying
        AWS service.
        """
    action = ServiceAction(action_model, factory=factory_self, service_context=service_context)
    if is_load:

        def do_action(self, *args, **kwargs):
            response = action(self, *args, **kwargs)
            self.meta.data = response
        lazy_docstring = docstring.LoadReloadDocstring(action_name=action_model.name, resource_name=resource_name, event_emitter=factory_self._emitter, load_model=action_model, service_model=service_context.service_model, include_signature=False)
    else:

        def do_action(self, *args, **kwargs):
            response = action(self, *args, **kwargs)
            if hasattr(self, 'load'):
                self.meta.data = None
            return response
        lazy_docstring = docstring.ActionDocstring(resource_name=resource_name, event_emitter=factory_self._emitter, action_model=action_model, service_model=service_context.service_model, include_signature=False)
    do_action.__name__ = str(action_model.name)
    do_action.__doc__ = lazy_docstring
    return do_action