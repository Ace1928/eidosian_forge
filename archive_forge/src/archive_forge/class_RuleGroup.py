from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.wafv2 import compare_priority_rules
from ansible_collections.community.aws.plugins.module_utils.wafv2 import describe_wafv2_tags
from ansible_collections.community.aws.plugins.module_utils.wafv2 import ensure_wafv2_tags
from ansible_collections.community.aws.plugins.module_utils.wafv2 import wafv2_list_rule_groups
from ansible_collections.community.aws.plugins.module_utils.wafv2 import wafv2_snake_dict_to_camel_dict
class RuleGroup:

    def __init__(self, wafv2, name, scope, fail_json_aws):
        self.wafv2 = wafv2
        self.id = None
        self.name = name
        self.scope = scope
        self.fail_json_aws = fail_json_aws
        self.existing_group = self.get_group()

    def update(self, description, rules, sampled_requests, cloudwatch_metrics, metric_name):
        req_obj = {'Name': self.name, 'Scope': self.scope, 'Id': self.id, 'Rules': rules, 'LockToken': self.locktoken, 'VisibilityConfig': {'SampledRequestsEnabled': sampled_requests, 'CloudWatchMetricsEnabled': cloudwatch_metrics, 'MetricName': metric_name}}
        if description:
            req_obj['Description'] = description
        try:
            response = self.wafv2.update_rule_group(**req_obj)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to update wafv2 rule group.')
        return self.refresh_group()

    def get_group(self):
        if self.id is None:
            response = self.list()
            for item in response.get('RuleGroups'):
                if item.get('Name') == self.name:
                    self.id = item.get('Id')
                    self.locktoken = item.get('LockToken')
                    self.arn = item.get('ARN')
        return self.refresh_group()

    def refresh_group(self):
        existing_group = None
        if self.id:
            try:
                response = self.wafv2.get_rule_group(Name=self.name, Scope=self.scope, Id=self.id)
                existing_group = response.get('RuleGroup')
                self.locktoken = response.get('LockToken')
            except (BotoCoreError, ClientError) as e:
                self.fail_json_aws(e, msg='Failed to get wafv2 rule group.')
            tags = describe_wafv2_tags(self.wafv2, self.arn, self.fail_json_aws)
            existing_group['tags'] = tags or {}
        return existing_group

    def list(self):
        return wafv2_list_rule_groups(self.wafv2, self.scope, self.fail_json_aws)

    def get(self):
        if self.existing_group:
            return self.existing_group
        return None

    def remove(self):
        try:
            response = self.wafv2.delete_rule_group(Name=self.name, Scope=self.scope, Id=self.id, LockToken=self.locktoken)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to delete wafv2 rule group.')
        return response

    def create(self, capacity, description, rules, sampled_requests, cloudwatch_metrics, metric_name, tags):
        req_obj = {'Name': self.name, 'Scope': self.scope, 'Capacity': capacity, 'Rules': rules, 'VisibilityConfig': {'SampledRequestsEnabled': sampled_requests, 'CloudWatchMetricsEnabled': cloudwatch_metrics, 'MetricName': metric_name}}
        if description:
            req_obj['Description'] = description
        if tags:
            req_obj['Tags'] = ansible_dict_to_boto3_tag_list(tags)
        try:
            response = self.wafv2.create_rule_group(**req_obj)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to create wafv2 rule group.')
        self.existing_group = self.get_group()
        return self.existing_group