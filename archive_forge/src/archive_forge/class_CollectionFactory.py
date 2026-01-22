import copy
import logging
from botocore import xform_name
from botocore.utils import merge_dicts
from .action import BatchAction
from .params import create_request_parameters
from .response import ResourceHandler
from ..docs import docstring
class CollectionFactory(object):
    """
    A factory to create new
    :py:class:`CollectionManager` and :py:class:`ResourceCollection`
    subclasses from a :py:class:`~boto3.resources.model.Collection`
    model. These subclasses include methods to perform batch operations.
    """

    def load_from_definition(self, resource_name, collection_model, service_context, event_emitter):
        """
        Loads a collection from a model, creating a new
        :py:class:`CollectionManager` subclass
        with the correct properties and methods, named based on the service
        and resource name, e.g. ec2.InstanceCollectionManager. It also
        creates a new :py:class:`ResourceCollection` subclass which is used
        by the new manager class.

        :type resource_name: string
        :param resource_name: Name of the resource to look up. For services,
                              this should match the ``service_name``.

        :type service_context: :py:class:`~boto3.utils.ServiceContext`
        :param service_context: Context about the AWS service

        :type event_emitter: :py:class:`~botocore.hooks.HierarchialEmitter`
        :param event_emitter: An event emitter

        :rtype: Subclass of :py:class:`CollectionManager`
        :return: The collection class.
        """
        attrs = {}
        collection_name = collection_model.name
        self._load_batch_actions(attrs, resource_name, collection_model, service_context.service_model, event_emitter)
        self._load_documented_collection_methods(attrs=attrs, resource_name=resource_name, collection_model=collection_model, service_model=service_context.service_model, event_emitter=event_emitter, base_class=ResourceCollection)
        if service_context.service_name == resource_name:
            cls_name = '{0}.{1}Collection'.format(service_context.service_name, collection_name)
        else:
            cls_name = '{0}.{1}.{2}Collection'.format(service_context.service_name, resource_name, collection_name)
        collection_cls = type(str(cls_name), (ResourceCollection,), attrs)
        self._load_documented_collection_methods(attrs=attrs, resource_name=resource_name, collection_model=collection_model, service_model=service_context.service_model, event_emitter=event_emitter, base_class=CollectionManager)
        attrs['_collection_cls'] = collection_cls
        cls_name += 'Manager'
        return type(str(cls_name), (CollectionManager,), attrs)

    def _load_batch_actions(self, attrs, resource_name, collection_model, service_model, event_emitter):
        """
        Batch actions on the collection become methods on both
        the collection manager and iterators.
        """
        for action_model in collection_model.batch_actions:
            snake_cased = xform_name(action_model.name)
            attrs[snake_cased] = self._create_batch_action(resource_name, snake_cased, action_model, collection_model, service_model, event_emitter)

    def _load_documented_collection_methods(factory_self, attrs, resource_name, collection_model, service_model, event_emitter, base_class):

        def all(self):
            return base_class.all(self)
        all.__doc__ = docstring.CollectionMethodDocstring(resource_name=resource_name, action_name='all', event_emitter=event_emitter, collection_model=collection_model, service_model=service_model, include_signature=False)
        attrs['all'] = all

        def filter(self, **kwargs):
            return base_class.filter(self, **kwargs)
        filter.__doc__ = docstring.CollectionMethodDocstring(resource_name=resource_name, action_name='filter', event_emitter=event_emitter, collection_model=collection_model, service_model=service_model, include_signature=False)
        attrs['filter'] = filter

        def limit(self, count):
            return base_class.limit(self, count)
        limit.__doc__ = docstring.CollectionMethodDocstring(resource_name=resource_name, action_name='limit', event_emitter=event_emitter, collection_model=collection_model, service_model=service_model, include_signature=False)
        attrs['limit'] = limit

        def page_size(self, count):
            return base_class.page_size(self, count)
        page_size.__doc__ = docstring.CollectionMethodDocstring(resource_name=resource_name, action_name='page_size', event_emitter=event_emitter, collection_model=collection_model, service_model=service_model, include_signature=False)
        attrs['page_size'] = page_size

    def _create_batch_action(factory_self, resource_name, snake_cased, action_model, collection_model, service_model, event_emitter):
        """
        Creates a new method which makes a batch operation request
        to the underlying service API.
        """
        action = BatchAction(action_model)

        def batch_action(self, *args, **kwargs):
            return action(self, *args, **kwargs)
        batch_action.__name__ = str(snake_cased)
        batch_action.__doc__ = docstring.BatchActionDocstring(resource_name=resource_name, event_emitter=event_emitter, batch_action_model=action_model, service_model=service_model, collection_model=collection_model, include_signature=False)
        return batch_action