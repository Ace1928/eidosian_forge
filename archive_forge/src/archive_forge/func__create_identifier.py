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
def _create_identifier(factory_self, identifier, resource_name):
    """
        Creates a read-only property for identifier attributes.
        """

    def get_identifier(self):
        return getattr(self, '_' + identifier.name, None)
    get_identifier.__name__ = str(identifier.name)
    get_identifier.__doc__ = docstring.IdentifierDocstring(resource_name=resource_name, identifier_model=identifier, include_signature=False)
    return property(get_identifier)