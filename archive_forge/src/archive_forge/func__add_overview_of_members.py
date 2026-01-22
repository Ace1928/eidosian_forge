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
def _add_overview_of_members(self, section):
    for resource_member_type in self.member_map:
        section.style.new_line()
        section.write("These are the resource's available %s:" % resource_member_type)
        section.style.new_line()
        for member in self.member_map[resource_member_type]:
            if resource_member_type in ['identifiers', 'attributes', 'references', 'collections']:
                section.style.li(':py:attr:`%s`' % member)
            else:
                section.style.li(':py:meth:`%s()`' % member)