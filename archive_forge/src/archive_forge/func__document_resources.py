import os
import boto3
from botocore.exceptions import DataNotFoundError
from botocore.docs.service import ServiceDocumenter as BaseServiceDocumenter
from botocore.docs.bcdoc.restdoc import DocumentStructure
from boto3.utils import ServiceContext
from boto3.docs.client import Boto3ClientDocumenter
from boto3.docs.resource import ResourceDocumenter
from boto3.docs.resource import ServiceResourceDocumenter
def _document_resources(self, section):
    temp_identifier_value = 'foo'
    loader = self._session.get_component('data_loader')
    json_resource_model = loader.load_service_model(self._service_name, 'resources-1')
    service_model = self._service_resource.meta.client.meta.service_model
    for resource_name in json_resource_model['resources']:
        resource_model = json_resource_model['resources'][resource_name]
        resource_cls = self._boto3_session.resource_factory.load_from_definition(resource_name=resource_name, single_resource_json_definition=resource_model, service_context=ServiceContext(service_name=self._service_name, resource_json_definitions=json_resource_model['resources'], service_model=service_model, service_waiter_model=None))
        identifiers = resource_cls.meta.resource_model.identifiers
        args = []
        for _ in identifiers:
            args.append(temp_identifier_value)
        resource = resource_cls(*args, client=self._client)
        ResourceDocumenter(resource, self._session).document_resource(section.add_new_section(resource.meta.resource_model.name))