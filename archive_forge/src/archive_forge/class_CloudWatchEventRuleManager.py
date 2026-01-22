import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
class CloudWatchEventRuleManager:
    RULE_FIELDS = ['name', 'event_pattern', 'schedule_expression', 'description', 'role_arn']

    def __init__(self, rule, targets):
        self.rule = rule
        self.targets = targets

    def ensure_present(self, enabled=True):
        """Ensures the rule and targets are present and synced"""
        rule_description = self.rule.describe()
        if rule_description:
            self._sync_rule(enabled)
            self._sync_targets()
            self._sync_state(enabled)
        else:
            self._create(enabled)

    def ensure_disabled(self):
        """Ensures the rule and targets are present, but disabled, and synced"""
        self.ensure_present(enabled=False)

    def ensure_absent(self):
        """Ensures the rule and targets are absent"""
        rule_description = self.rule.describe()
        if not rule_description:
            return
        self.rule.delete()

    def fetch_aws_state(self):
        """Retrieves rule and target state from AWS"""
        aws_state = {'rule': {}, 'targets': [], 'changed': self.rule.changed}
        rule_description = self.rule.describe()
        if not rule_description:
            return aws_state
        del rule_description['response_metadata']
        aws_state['rule'] = rule_description
        aws_state['targets'].extend(self.rule.list_targets())
        return aws_state

    def _sync_rule(self, enabled=True):
        """Syncs local rule state with AWS"""
        if not self._rule_matches_aws():
            self.rule.put(enabled)

    def _sync_targets(self):
        """Syncs local targets with AWS"""
        target_ids_to_remove = self._remote_target_ids_to_remove()
        if target_ids_to_remove:
            self.rule.remove_targets(target_ids_to_remove)
        targets_to_put = self._targets_to_put()
        if targets_to_put:
            self.rule.put_targets(targets_to_put)

    def _sync_state(self, enabled=True):
        """Syncs local rule state with AWS"""
        remote_state = self._remote_state()
        if enabled and remote_state != 'ENABLED':
            self.rule.enable()
        elif not enabled and remote_state != 'DISABLED':
            self.rule.disable()

    def _create(self, enabled=True):
        """Creates rule and targets on AWS"""
        self.rule.put(enabled)
        self.rule.put_targets(self.targets)

    def _rule_matches_aws(self):
        """Checks if the local rule data matches AWS"""
        aws_rule_data = self.rule.describe()
        return all((getattr(self.rule, field) == aws_rule_data.get(field, None) for field in self.RULE_FIELDS))

    def _targets_to_put(self):
        """Returns a list of targets that need to be updated or added remotely"""
        remote_targets = self.rule.list_targets()
        temp = []
        for t in self.targets:
            if t['input_transformer'] is not None and t['input_transformer']['input_template'] is not None:
                val = t['input_transformer']['input_template']
                valid_json = _validate_json(val)
                if not valid_json:
                    t['input_transformer']['input_template'] = '"' + val + '"'
            temp.append(scrub_none_parameters(t))
        self.targets = temp
        return [t for t in self.targets if camel_dict_to_snake_dict(t) not in remote_targets]

    def _remote_target_ids_to_remove(self):
        """Returns a list of targets that need to be removed remotely"""
        target_ids = [t['id'] for t in self.targets]
        remote_targets = self.rule.list_targets()
        return [rt['id'] for rt in remote_targets if rt['id'] not in target_ids]

    def _remote_state(self):
        """Returns the remote state from AWS"""
        description = self.rule.describe()
        if not description:
            return
        return description['state']