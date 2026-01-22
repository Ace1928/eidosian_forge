import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
class CloudFormationServiceManager:
    """Handles CloudFormation Services"""

    def __init__(self, module):
        self.module = module
        self.client = module.client('cloudformation')

    @AWSRetry.exponential_backoff(retries=5, delay=5)
    def describe_stacks_with_backoff(self, **kwargs):
        paginator = self.client.get_paginator('describe_stacks')
        return paginator.paginate(**kwargs).build_full_result()['Stacks']

    def describe_stacks(self, stack_name=None):
        try:
            kwargs = {'StackName': stack_name} if stack_name else {}
            response = self.describe_stacks_with_backoff(**kwargs)
            if response is not None:
                return response
            self.module.fail_json(msg='Error describing stack(s) - an empty response was returned')
        except is_boto3_error_message('does not exist'):
            return {}
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Error describing stack ' + stack_name)

    @AWSRetry.exponential_backoff(retries=5, delay=5)
    def list_stack_resources_with_backoff(self, stack_name):
        paginator = self.client.get_paginator('list_stack_resources')
        return paginator.paginate(StackName=stack_name).build_full_result()['StackResourceSummaries']

    def list_stack_resources(self, stack_name):
        try:
            return self.list_stack_resources_with_backoff(stack_name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Error listing stack resources for stack ' + stack_name)

    @AWSRetry.exponential_backoff(retries=5, delay=5)
    def describe_stack_events_with_backoff(self, stack_name):
        paginator = self.client.get_paginator('describe_stack_events')
        return paginator.paginate(StackName=stack_name).build_full_result()['StackEvents']

    def describe_stack_events(self, stack_name):
        try:
            return self.describe_stack_events_with_backoff(stack_name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Error listing stack events for stack ' + stack_name)

    @AWSRetry.exponential_backoff(retries=5, delay=5)
    def list_stack_change_sets_with_backoff(self, stack_name):
        paginator = self.client.get_paginator('list_change_sets')
        return paginator.paginate(StackName=stack_name).build_full_result()['Summaries']

    @AWSRetry.exponential_backoff(retries=5, delay=5)
    def describe_stack_change_set_with_backoff(self, **kwargs):
        paginator = self.client.get_paginator('describe_change_set')
        return paginator.paginate(**kwargs).build_full_result()

    def describe_stack_change_sets(self, stack_name):
        changes = []
        try:
            change_sets = self.list_stack_change_sets_with_backoff(stack_name)
            for item in change_sets:
                changes.append(self.describe_stack_change_set_with_backoff(StackName=stack_name, ChangeSetName=item['ChangeSetName']))
            return changes
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Error describing stack change sets for stack ' + stack_name)

    @AWSRetry.exponential_backoff(retries=5, delay=5)
    def get_stack_policy_with_backoff(self, stack_name):
        return self.client.get_stack_policy(StackName=stack_name)

    def get_stack_policy(self, stack_name):
        try:
            response = self.get_stack_policy_with_backoff(stack_name)
            stack_policy = response.get('StackPolicyBody')
            if stack_policy:
                return json.loads(stack_policy)
            return dict()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Error getting stack policy for stack ' + stack_name)

    @AWSRetry.exponential_backoff(retries=5, delay=5)
    def get_template_with_backoff(self, stack_name):
        return self.client.get_template(StackName=stack_name)

    def get_template(self, stack_name):
        try:
            response = self.get_template_with_backoff(stack_name)
            return response.get('TemplateBody')
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Error getting stack template for stack ' + stack_name)