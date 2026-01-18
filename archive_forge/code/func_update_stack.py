import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def update_stack(self, stack_name, template_body=None, template_url=None, parameters=None, notification_arns=None, disable_rollback=False, timeout_in_minutes=None, capabilities=None, tags=None, use_previous_template=None, stack_policy_during_update_body=None, stack_policy_during_update_url=None, stack_policy_body=None, stack_policy_url=None):
    """
        Updates a stack as specified in the template. After the call
        completes successfully, the stack update starts. You can check
        the status of the stack via the DescribeStacks action.



        **Note: **You cannot update `AWS::S3::Bucket`_ resources, for
        example, to add or modify tags.



        To get a copy of the template for an existing stack, you can
        use the GetTemplate action.

        Tags that were associated with this stack during creation time
        will still be associated with the stack after an `UpdateStack`
        operation.

        For more information about creating an update template,
        updating a stack, and monitoring the progress of the update,
        see `Updating a Stack`_.

        :type stack_name: string
        :param stack_name:
        The name or stack ID of the stack to update.

        Must contain only alphanumeric characters (case sensitive) and start
            with an alpha character. Maximum length of the name is 255
            characters.

        :type template_body: string
        :param template_body: Structure containing the template body. (For more
            information, go to `Template Anatomy`_ in the AWS CloudFormation
            User Guide.)
        Conditional: You must pass either `UsePreviousTemplate` or one of
            `TemplateBody` or `TemplateUrl`. If both `TemplateBody` and
            `TemplateUrl` are passed, only `TemplateBody` is used.

        :type template_url: string
        :param template_url: Location of file containing the template body. The
            URL must point to a template (max size: 307,200 bytes) located in
            an S3 bucket in the same region as the stack. For more information,
            go to the `Template Anatomy`_ in the AWS CloudFormation User Guide.
        Conditional: You must pass either `UsePreviousTemplate` or one of
            `TemplateBody` or `TemplateUrl`. If both `TemplateBody` and
            `TemplateUrl` are passed, only `TemplateBody` is used.
            `TemplateBody`.

        :type use_previous_template: boolean
        :param use_previous_template: Set to `True` to use the previous
            template instead of uploading a new one via `TemplateBody` or
            `TemplateURL`.
        Conditional: You must pass either `UsePreviousTemplate` or one of
            `TemplateBody` or `TemplateUrl`.

        :type parameters: list
        :param parameters: A list of key/value tuples that specify input
            parameters for the stack. A 3-tuple (key, value, bool) may be used to
            specify the `UsePreviousValue` option.

        :type notification_arns: list
        :param notification_arns: The Simple Notification Service (SNS) topic
            ARNs to publish stack related events. You can find your SNS topic
            ARNs using the `SNS console`_ or your Command Line Interface (CLI).

        :type disable_rollback: bool
        :param disable_rollback: Indicates whether or not to rollback on
            failure.

        :type timeout_in_minutes: integer
        :param timeout_in_minutes: The amount of time that can pass before the
            stack status becomes CREATE_FAILED; if `DisableRollback` is not set
            or is set to `False`, the stack will be rolled back.

        :type capabilities: list
        :param capabilities: The list of capabilities you want to allow in
            the stack.  Currently, the only valid capability is
            'CAPABILITY_IAM'.

        :type tags: dict
        :param tags: A set of user-defined `Tags` to associate with this stack,
            represented by key/value pairs. Tags defined for the stack are
            propagated to EC2 resources that are created as part of the stack.
            A maximum number of 10 tags can be specified.

        :type template_url: string
        :param template_url: Location of file containing the template body. The
            URL must point to a template located in an S3 bucket in the same
            region as the stack. For more information, go to `Template
            Anatomy`_ in the AWS CloudFormation User Guide.
        Conditional: You must pass `TemplateURL` or `TemplateBody`. If both are
            passed, only `TemplateBody` is used.

        :type stack_policy_during_update_body: string
        :param stack_policy_during_update_body: Structure containing the
            temporary overriding stack policy body. If you pass
            `StackPolicyDuringUpdateBody` and `StackPolicyDuringUpdateURL`,
            only `StackPolicyDuringUpdateBody` is used.
        If you want to update protected resources, specify a temporary
            overriding stack policy during this update. If you do not specify a
            stack policy, the current policy that associated with the stack
            will be used.

        :type stack_policy_during_update_url: string
        :param stack_policy_during_update_url: Location of a file containing
            the temporary overriding stack policy. The URL must point to a
            policy (max size: 16KB) located in an S3 bucket in the same region
            as the stack. If you pass `StackPolicyDuringUpdateBody` and
            `StackPolicyDuringUpdateURL`, only `StackPolicyDuringUpdateBody` is
            used.
        If you want to update protected resources, specify a temporary
            overriding stack policy during this update. If you do not specify a
            stack policy, the current policy that is associated with the stack
            will be used.

        :rtype: string
        :return: The unique Stack ID.
        """
    params = self._build_create_or_update_params(stack_name, template_body, template_url, parameters, disable_rollback, timeout_in_minutes, notification_arns, capabilities, None, stack_policy_body, stack_policy_url, tags, use_previous_template, stack_policy_during_update_body, stack_policy_during_update_url)
    body = self._do_request('UpdateStack', params, '/', 'POST')
    return body['UpdateStackResponse']['UpdateStackResult']['StackId']