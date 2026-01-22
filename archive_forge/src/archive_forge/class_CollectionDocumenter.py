from botocore import xform_name
from botocore.docs.method import get_instance_public_methods
from botocore.docs.utils import DocumentedShape
from boto3.docs.base import BaseDocumenter
from boto3.docs.utils import get_resource_ignore_params
from boto3.docs.method import document_model_driven_resource_method
from boto3.docs.utils import add_resource_type_overview
class CollectionDocumenter(BaseDocumenter):

    def document_collections(self, section):
        collections = self._resource.meta.resource_model.collections
        collections_list = []
        add_resource_type_overview(section=section, resource_type='Collections', description='Collections provide an interface to iterate over and manipulate groups of resources. ', intro_link='guide_collections')
        self.member_map['collections'] = collections_list
        for collection in collections:
            collection_section = section.add_new_section(collection.name)
            collections_list.append(collection.name)
            self._document_collection(collection_section, collection)

    def _document_collection(self, section, collection):
        methods = get_instance_public_methods(getattr(self._resource, collection.name))
        document_collection_object(section, collection)
        batch_actions = {}
        for batch_action in collection.batch_actions:
            batch_actions[batch_action.name] = batch_action
        for method in sorted(methods):
            method_section = section.add_new_section(method)
            if method in batch_actions:
                document_batch_action(section=method_section, resource_name=self._resource_name, event_emitter=self._resource.meta.client.meta.events, batch_action_model=batch_actions[method], collection_model=collection, service_model=self._resource.meta.client.meta.service_model)
            else:
                document_collection_method(section=method_section, resource_name=self._resource_name, action_name=method, event_emitter=self._resource.meta.client.meta.events, collection_model=collection, service_model=self._resource.meta.client.meta.service_model)