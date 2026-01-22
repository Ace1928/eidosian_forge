import traceback
from copy import deepcopy
from .ec2 import get_ec2_security_group_ids_from_names
from .elb_utils import convert_tg_name_to_arn
from .elb_utils import get_elb
from .elb_utils import get_elb_listener
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .waiters import get_waiter
class ELBListenerRule:

    def __init__(self, connection, module, rule, listener_arn):
        self.connection = connection
        self.module = module
        self.rule = rule
        self.listener_arn = listener_arn
        self.changed = False

    def create(self):
        """
        Create a listener rule

        :return:
        """
        try:
            self.rule['ListenerArn'] = self.listener_arn
            self.rule['Priority'] = int(self.rule['Priority'])
            AWSRetry.jittered_backoff()(self.connection.create_rule)(**self.rule)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e)
        self.changed = True

    def modify(self):
        """
        Modify a listener rule

        :return:
        """
        try:
            del self.rule['Priority']
            AWSRetry.jittered_backoff()(self.connection.modify_rule)(**self.rule)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e)
        self.changed = True

    def delete(self):
        """
        Delete a listener rule

        :return:
        """
        try:
            AWSRetry.jittered_backoff()(self.connection.delete_rule)(RuleArn=self.rule['RuleArn'])
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e)
        self.changed = True

    def set_rule_priorities(self):
        """
        Sets the priorities of the specified rules.

        :return:
        """
        try:
            rules = [self.rule]
            if isinstance(self.rule, list):
                rules = self.rule
            rule_priorities = [{'RuleArn': rule['RuleArn'], 'Priority': rule['Priority']} for rule in rules]
            AWSRetry.jittered_backoff()(self.connection.set_rule_priorities)(RulePriorities=rule_priorities)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e)
        self.changed = True