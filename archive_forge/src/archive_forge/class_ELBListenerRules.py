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
class ELBListenerRules:

    def __init__(self, connection, module, elb_arn, listener_rules, listener_port):
        self.connection = connection
        self.module = module
        self.elb_arn = elb_arn
        self.rules = self._ensure_rules_action_has_arn(listener_rules)
        self.changed = False
        self.current_listener = get_elb_listener(connection, module, elb_arn, listener_port)
        self.listener_arn = self.current_listener.get('ListenerArn')
        if 'ListenerArn' in self.current_listener:
            self.current_rules = self._get_elb_listener_rules()
        else:
            self.current_rules = []

    def _ensure_rules_action_has_arn(self, rules):
        """
        If a rule Action has been passed with a Target Group Name instead of ARN, lookup the ARN and
        replace the name.

        :param rules: a list of rule dicts
        :return: the same list of dicts ensuring that each rule Actions dict has TargetGroupArn key. If a TargetGroupName key exists, it is removed.
        """
        fixed_rules = []
        for rule in rules:
            fixed_actions = []
            for action in rule['Actions']:
                if 'TargetGroupName' in action:
                    action['TargetGroupArn'] = convert_tg_name_to_arn(self.connection, self.module, action['TargetGroupName'])
                    del action['TargetGroupName']
                fixed_actions.append(action)
            rule['Actions'] = fixed_actions
            fixed_rules.append(rule)
        return fixed_rules

    def _get_elb_listener_rules(self):
        try:
            return AWSRetry.jittered_backoff()(self.connection.describe_rules)(ListenerArn=self.current_listener['ListenerArn'])['Rules']
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e)

    def _compare_condition(self, current_conditions, condition):
        """

        :param current_conditions:
        :param condition:
        :return:
        """
        condition_found = False
        for current_condition in current_conditions:
            if current_condition.get('HostHeaderConfig') and condition.get('HostHeaderConfig'):
                if current_condition['Field'] == condition['Field'] and sorted(current_condition['HostHeaderConfig']['Values']) == sorted(condition['HostHeaderConfig']['Values']):
                    condition_found = True
                    break
            elif current_condition.get('HttpHeaderConfig'):
                if current_condition['Field'] == condition['Field'] and sorted(current_condition['HttpHeaderConfig']['Values']) == sorted(condition['HttpHeaderConfig']['Values']) and (current_condition['HttpHeaderConfig']['HttpHeaderName'] == condition['HttpHeaderConfig']['HttpHeaderName']):
                    condition_found = True
                    break
            elif current_condition.get('HttpRequestMethodConfig'):
                if current_condition['Field'] == condition['Field'] and sorted(current_condition['HttpRequestMethodConfig']['Values']) == sorted(condition['HttpRequestMethodConfig']['Values']):
                    condition_found = True
                    break
            elif current_condition.get('PathPatternConfig') and condition.get('PathPatternConfig'):
                if current_condition['Field'] == condition['Field'] and sorted(current_condition['PathPatternConfig']['Values']) == sorted(condition['PathPatternConfig']['Values']):
                    condition_found = True
                    break
            elif current_condition.get('QueryStringConfig'):
                if current_condition['Field'] == condition['Field'] and current_condition['QueryStringConfig']['Values'] == condition['QueryStringConfig']['Values']:
                    condition_found = True
                    break
            elif current_condition.get('SourceIpConfig'):
                if current_condition['Field'] == condition['Field'] and sorted(current_condition['SourceIpConfig']['Values']) == sorted(condition['SourceIpConfig']['Values']):
                    condition_found = True
                    break
            elif current_condition['Field'] == condition['Field'] and sorted(current_condition['Values']) == sorted(condition['Values']):
                condition_found = True
                break
        return condition_found

    def _compare_rule(self, current_rule, new_rule):
        """

        :return:
        """
        modified_rule = {}
        if int(current_rule['Priority']) != int(new_rule['Priority']):
            modified_rule['Priority'] = new_rule['Priority']
        if len(current_rule['Actions']) == len(new_rule['Actions']):
            copy_new_rule = deepcopy(new_rule)
            current_actions_sorted = _sort_actions(current_rule['Actions'])
            new_actions_sorted = _sort_actions(copy_new_rule['Actions'])
            new_current_actions_sorted = [_append_use_existing_client_secretn(i) for i in current_actions_sorted]
            new_actions_sorted_no_secret = [_prune_secret(i) for i in new_actions_sorted]
            if [_prune_ForwardConfig(i) for i in new_current_actions_sorted] != [_prune_ForwardConfig(i) for i in new_actions_sorted_no_secret]:
                modified_rule['Actions'] = new_rule['Actions']
        else:
            modified_rule['Actions'] = new_rule['Actions']
        modified_conditions = []
        for condition in new_rule['Conditions']:
            if not self._compare_condition(current_rule['Conditions'], condition):
                modified_conditions.append(condition)
        if modified_conditions:
            modified_rule['Conditions'] = modified_conditions
        return modified_rule

    def compare_rules(self):
        """

        :return:
        """
        rules_to_modify = []
        rules_to_delete = []
        rules_to_add = deepcopy(self.rules)
        rules_to_set_priority = []
        current_rules = deepcopy(self.current_rules)
        remaining_rules = []
        while current_rules:
            current_rule = current_rules.pop(0)
            if current_rule.get('IsDefault', False):
                continue
            to_keep = True
            for new_rule in rules_to_add:
                modified_rule = self._compare_rule(current_rule, new_rule)
                if not modified_rule:
                    rules_to_add.remove(new_rule)
                    to_keep = False
                    break
                if modified_rule and list(modified_rule.keys()) == ['Priority']:
                    modified_rule['Priority'] = int(new_rule['Priority'])
                    modified_rule['RuleArn'] = current_rule['RuleArn']
                    rules_to_set_priority.append(modified_rule)
                    to_keep = False
                    rules_to_add.remove(new_rule)
                    break
            if to_keep:
                remaining_rules.append(current_rule)
        for current_rule in remaining_rules:
            current_rule_passed_to_module = False
            for new_rule in rules_to_add:
                if current_rule['Priority'] == str(new_rule['Priority']):
                    current_rule_passed_to_module = True
                    rules_to_add.remove(new_rule)
                    modified_rule = self._compare_rule(current_rule, new_rule)
                    if modified_rule:
                        modified_rule['Priority'] = int(current_rule['Priority'])
                        modified_rule['RuleArn'] = current_rule['RuleArn']
                        modified_rule['Actions'] = new_rule['Actions']
                        modified_rule['Conditions'] = new_rule['Conditions']
                        for action in modified_rule.get('Actions', []):
                            if action.get('AuthenticateOidcConfig', {}).get('ClientSecret', False):
                                action['AuthenticateOidcConfig']['UseExistingClientSecret'] = False
                        rules_to_modify.append(modified_rule)
                    break
            if not current_rule_passed_to_module and (not current_rule.get('IsDefault', False)):
                rules_to_delete.append(current_rule['RuleArn'])
        for rule in rules_to_add:
            for action in rule.get('Actions', []):
                if action.get('AuthenticateOidcConfig', {}).get('UseExistingClientSecret', False):
                    action['AuthenticateOidcConfig']['UseExistingClientSecret'] = False
        return (rules_to_add, rules_to_modify, rules_to_delete, rules_to_set_priority)