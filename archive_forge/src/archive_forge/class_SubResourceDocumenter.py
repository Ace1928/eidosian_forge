from botocore import xform_name
from botocore.utils import get_service_module_name
from boto3.docs.base import BaseDocumenter
from boto3.docs.utils import get_identifier_args_for_signature
from boto3.docs.utils import get_identifier_values_for_example
from boto3.docs.utils import get_identifier_description
from boto3.docs.utils import add_resource_type_overview
class SubResourceDocumenter(BaseDocumenter):

    def document_sub_resources(self, section):
        add_resource_type_overview(section=section, resource_type='Sub-resources', description="Sub-resources are methods that create a new instance of a child resource. This resource's identifiers get passed along to the child.", intro_link='subresources_intro')
        sub_resources = sorted(self._resource.meta.resource_model.subresources, key=lambda sub_resource: sub_resource.name)
        sub_resources_list = []
        self.member_map['sub-resources'] = sub_resources_list
        for sub_resource in sub_resources:
            sub_resource_section = section.add_new_section(sub_resource.name)
            sub_resources_list.append(sub_resource.name)
            document_sub_resource(section=sub_resource_section, resource_name=self._resource_name, sub_resource_model=sub_resource, service_model=self._service_model)