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
def _create_class_partial(factory_self, subresource_model, resource_name, service_context):
    """
        Creates a new method which acts as a functools.partial, passing
        along the instance's low-level `client` to the new resource
        class' constructor.
        """
    name = subresource_model.resource.type

    def create_resource(self, *args, **kwargs):
        positional_args = []
        json_def = service_context.resource_json_definitions.get(name, {})
        resource_cls = factory_self.load_from_definition(resource_name=name, single_resource_json_definition=json_def, service_context=service_context)
        identifiers = subresource_model.resource.identifiers
        if identifiers is not None:
            for identifier, value in build_identifiers(identifiers, self):
                positional_args.append(value)
        return partial(resource_cls, *positional_args, client=self.meta.client)(*args, **kwargs)
    create_resource.__name__ = str(name)
    create_resource.__doc__ = docstring.SubResourceDocstring(resource_name=resource_name, sub_resource_model=subresource_model, service_model=service_context.service_model, include_signature=False)
    return create_resource