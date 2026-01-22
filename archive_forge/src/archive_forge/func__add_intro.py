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
def _add_intro(self, section):
    identifier_names = []
    if self._resource_model.identifiers:
        for identifier in self._resource_model.identifiers:
            identifier_names.append(identifier.name)
    class_args = get_identifier_args_for_signature(identifier_names)
    section.style.start_sphinx_py_class(class_name='%s(%s)' % (self.class_name, class_args))
    description_section = section.add_new_section('description')
    self._add_description(description_section)
    example_section = section.add_new_section('example')
    self._add_example(example_section, identifier_names)
    param_section = section.add_new_section('params')
    self._add_params_description(param_section, identifier_names)