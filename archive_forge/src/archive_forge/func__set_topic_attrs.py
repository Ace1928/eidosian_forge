import json
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.sns import canonicalize_endpoint
from ansible_collections.community.aws.plugins.module_utils.sns import compare_delivery_policies
from ansible_collections.community.aws.plugins.module_utils.sns import get_info
from ansible_collections.community.aws.plugins.module_utils.sns import list_topic_subscriptions
from ansible_collections.community.aws.plugins.module_utils.sns import list_topics
from ansible_collections.community.aws.plugins.module_utils.sns import topic_arn_lookup
from ansible_collections.community.aws.plugins.module_utils.sns import update_tags
def _set_topic_attrs(self):
    changed = False
    try:
        topic_attributes = self.connection.get_topic_attributes(TopicArn=self.topic_arn)['Attributes']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg=f"Couldn't get topic attributes for topic {self.topic_arn}")
    if self.display_name and self.display_name != topic_attributes['DisplayName']:
        changed = True
        self.attributes_set.append('display_name')
        if not self.check_mode:
            try:
                self.connection.set_topic_attributes(TopicArn=self.topic_arn, AttributeName='DisplayName', AttributeValue=self.display_name)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg="Couldn't set display name")
    if self.policy and compare_policies(self.policy, json.loads(topic_attributes['Policy'])):
        changed = True
        self.attributes_set.append('policy')
        if not self.check_mode:
            try:
                self.connection.set_topic_attributes(TopicArn=self.topic_arn, AttributeName='Policy', AttributeValue=json.dumps(self.policy))
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg="Couldn't set topic policy")
    if ('FifoTopic' in topic_attributes and topic_attributes['FifoTopic'] == 'true') and self.content_based_deduplication:
        enabled = 'true' if self.content_based_deduplication in 'enabled' else 'false'
        if enabled != topic_attributes['ContentBasedDeduplication']:
            changed = True
            self.attributes_set.append('content_based_deduplication')
            if not self.check_mode:
                try:
                    self.connection.set_topic_attributes(TopicArn=self.topic_arn, AttributeName='ContentBasedDeduplication', AttributeValue=enabled)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg="Couldn't set content-based deduplication")
    if self.delivery_policy and ('DeliveryPolicy' not in topic_attributes or compare_delivery_policies(self.delivery_policy, json.loads(topic_attributes['DeliveryPolicy']))):
        changed = True
        self.attributes_set.append('delivery_policy')
        if not self.check_mode:
            try:
                self.connection.set_topic_attributes(TopicArn=self.topic_arn, AttributeName='DeliveryPolicy', AttributeValue=json.dumps(self.delivery_policy))
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg="Couldn't set topic delivery policy")
    return changed