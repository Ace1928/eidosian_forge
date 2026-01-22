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
def _load_identifiers(self, attrs, meta, resource_model, resource_name):
    """
        Populate required identifiers. These are arguments without which
        the resource cannot be used. Identifiers become arguments for
        operations on the resource.
        """
    for identifier in resource_model.identifiers:
        meta.identifiers.append(identifier.name)
        attrs[identifier.name] = self._create_identifier(identifier, resource_name)