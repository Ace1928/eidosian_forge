import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
class NFRuleGroupBoto3Mixin(NetworkFirewallBoto3Mixin):

    @AWSRetry.jittered_backoff()
    def _paginated_list_rule_groups(self, **params):
        paginator = self.client.get_paginator('list_rule_groups')
        result = paginator.paginate(**params).build_full_result()
        return result.get('RuleGroups', None)

    @Boto3Mixin.aws_error_handler('list all rule groups')
    def _list_rule_groups(self, **params):
        return self._paginated_list_rule_groups(**params)

    @Boto3Mixin.aws_error_handler('describe rule group')
    def _describe_rule_group(self, **params):
        try:
            result = self.client.describe_rule_group(aws_retry=True, **params)
        except is_boto3_error_code('ResourceNotFoundException'):
            return None
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        rule_group = result.get('RuleGroup', None)
        metadata = result.get('RuleGroupResponse', None)
        return dict(RuleGroup=rule_group, RuleGroupMetadata=metadata)

    @Boto3Mixin.aws_error_handler('create rule group')
    def _create_rule_group(self, **params):
        result = self.client.create_rule_group(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('RuleGroupResponse', None)

    @Boto3Mixin.aws_error_handler('update rule group')
    def _update_rule_group(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.update_rule_group(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('RuleGroupResponse', None)

    @Boto3Mixin.aws_error_handler('delete rule group')
    def _delete_rule_group(self, **params):
        try:
            result = self.client.delete_rule_group(aws_retry=True, **params)
        except is_boto3_error_code('ResourceNotFoundException'):
            return None
        return result.get('RuleGroupResponse', None)

    @Boto3Mixin.aws_error_handler('firewall rule to finish deleting')
    def _wait_rule_group_deleted(self, **params):
        waiter = self.nf_waiter_factory.get_waiter('rule_group_deleted')
        waiter.wait(**params)

    @Boto3Mixin.aws_error_handler('firewall rule to become active')
    def _wait_rule_group_active(self, **params):
        waiter = self.nf_waiter_factory.get_waiter('rule_group_active')
        waiter.wait(**params)