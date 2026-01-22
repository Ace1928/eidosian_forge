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
def _create_identifier_alias(factory_self, resource_name, identifier, member_model, service_context):
    """
        Creates a read-only property that aliases an identifier.
        """

    def get_identifier(self):
        return getattr(self, '_' + identifier.name, None)
    get_identifier.__name__ = str(identifier.member_name)
    get_identifier.__doc__ = docstring.AttributeDocstring(service_name=service_context.service_name, resource_name=resource_name, attr_name=identifier.member_name, event_emitter=factory_self._emitter, attr_model=member_model, include_signature=False)
    return property(get_identifier)