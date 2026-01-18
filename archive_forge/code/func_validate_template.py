import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def validate_template(self, template_body=None, template_url=None):
    """
        Validates a specified template.

        :type template_body: string
        :param template_body: String containing the template body. (For more
            information, go to `Template Anatomy`_ in the AWS CloudFormation
            User Guide.)
        Conditional: You must pass `TemplateURL` or `TemplateBody`. If both are
            passed, only `TemplateBody` is used.

        :type template_url: string
        :param template_url: Location of file containing the template body. The
            URL must point to a template (max size: 307,200 bytes) located in
            an S3 bucket in the same region as the stack. For more information,
            go to `Template Anatomy`_ in the AWS CloudFormation User Guide.
        Conditional: You must pass `TemplateURL` or `TemplateBody`. If both are
            passed, only `TemplateBody` is used.

        """
    params = {}
    if template_body:
        params['TemplateBody'] = template_body
    if template_url:
        params['TemplateURL'] = template_url
    if template_body and template_url:
        boto.log.warning('If both TemplateBody and TemplateURL are specified, only TemplateBody will be honored by the API')
    return self.get_object('ValidateTemplate', params, Template, verb='POST')