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
def _set_topic_subs(self):
    changed = False
    subscriptions_existing_list = set()
    desired_subscriptions = [(sub['protocol'], canonicalize_endpoint(sub['protocol'], sub['endpoint'])) for sub in self.subscriptions]
    for sub in list_topic_subscriptions(self.connection, self.module, self.topic_arn):
        sub_key = (sub['Protocol'], sub['Endpoint'])
        subscriptions_existing_list.add(sub_key)
        if self.purge_subscriptions and sub_key not in desired_subscriptions and (sub['SubscriptionArn'] not in ('PendingConfirmation', 'Deleted')):
            changed = True
            self.subscriptions_deleted.append(sub_key)
            if not self.check_mode:
                try:
                    self.connection.unsubscribe(SubscriptionArn=sub['SubscriptionArn'])
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg="Couldn't unsubscribe from topic")
    for protocol, endpoint in set(desired_subscriptions).difference(subscriptions_existing_list):
        changed = True
        self.subscriptions_added.append((protocol, endpoint))
        if not self.check_mode:
            try:
                self.connection.subscribe(TopicArn=self.topic_arn, Protocol=protocol, Endpoint=endpoint)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg=f"Couldn't subscribe to topic {self.topic_arn}")
    return changed