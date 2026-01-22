import os
from botocore import xform_name
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.utils import get_official_service_name
from boto3.docs.action import ActionDocumenter
from boto3.docs.attr import (
from boto3.docs.base import BaseDocumenter
from boto3.docs.collection import CollectionDocumenter
from boto3.docs.subresource import SubResourceDocumenter
from boto3.docs.utils import (
from boto3.docs.waiter import WaiterResourceDocumenter
def _add_overview_of_member_type(self, section, resource_member_type):
    section.style.new_line()
    section.write(f"These are the resource's available {resource_member_type}:")
    section.style.new_line()
    section.style.toctree()
    for member in self.member_map[resource_member_type]:
        section.style.tocitem(f'{member}')