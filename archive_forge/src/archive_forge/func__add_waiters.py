from botocore import xform_name
from botocore.docs.utils import get_official_service_name
from boto3.docs.base import BaseDocumenter
from boto3.docs.action import ActionDocumenter
from boto3.docs.waiter import WaiterResourceDocumenter
from boto3.docs.collection import CollectionDocumenter
from boto3.docs.subresource import SubResourceDocumenter
from boto3.docs.attr import document_attribute
from boto3.docs.attr import document_identifier
from boto3.docs.attr import document_reference
from boto3.docs.utils import get_identifier_args_for_signature
from boto3.docs.utils import get_identifier_values_for_example
from boto3.docs.utils import get_identifier_description
from boto3.docs.utils import add_resource_type_overview
def _add_waiters(self, section):
    section = section.add_new_section('waiters')
    waiters = self._resource.meta.resource_model.waiters
    if waiters:
        service_waiter_model = self._botocore_session.get_waiter_model(self._service_name)
        documenter = WaiterResourceDocumenter(self._resource, service_waiter_model)
        documenter.member_map = self.member_map
        documenter.document_resource_waiters(section)