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
class NetworkFirewallPolicyManager(NFPolicyBoto3Mixin, NFRuleGroupBoto3Mixin, BaseNetworkFirewallManager):
    name = None
    arn = None
    _group_name_cache = None

    def __init__(self, module, name=None, arn=None):
        super().__init__(module)
        self.name = name
        self.arn = arn
        if self.name or self.arn:
            policy = deepcopy(self.get_policy())
            self.original_resource = policy

    def _extra_error_output(self):
        output = super(NetworkFirewallPolicyManager, self)._extra_error_output()
        if self.name:
            output['FirewallPolicyName'] = self.name
        if self.arn:
            output['FirewallPolicyArn'] = self.arn
        return output

    def _filter_immutable_metadata_attributes(self, metadata):
        metadata = super(NetworkFirewallPolicyManager, self)._filter_immutable_metadata_attributes(metadata)
        metadata.pop('FirewallPolicyArn', None)
        metadata.pop('FirewallPolicyName', None)
        metadata.pop('FirewallPolicyId', None)
        metadata.pop('FirewallPolicyStatus', None)
        metadata.pop('ConsumedStatelessRuleCapacity', None)
        metadata.pop('ConsumedStatefulRuleCapacity', None)
        metadata.pop('Tags', None)
        metadata.pop('NumberOfAssociations', None)
        return metadata

    def _get_preupdate_arn(self):
        return self._get_metadata_value('FirewallPolicyArn')

    def _get_id_params(self, name=None, arn=None):
        if arn:
            return dict(FirewallPolicyArn=arn)
        if self.arn:
            return dict(FirewallPolicyArn=self.arn)
        if not name:
            name = self.name
        return dict(FirewallPolicyName=name)

    def delete(self, name=None, arn=None):
        id_params = self._get_id_params(name=name, arn=arn)
        result = self._get_policy(**id_params)
        if not result:
            return False
        self.updated_resource = dict()
        rule_status = self._get_metadata_value('FirewallPolicyStatus', '').upper()
        if rule_status == 'DELETING':
            self._wait_for_deletion()
            return False
        if self.module.check_mode:
            self.changed = True
            return True
        result = self._delete_policy(**id_params)
        self._wait_for_deletion()
        self.changed |= bool(result)
        return bool(result)

    def list(self):
        params = dict()
        policies = self._list_policies(**params)
        if not policies:
            return list()
        return [p.get('Arn', None) for p in policies]

    @property
    def _rule_group_name_cache(self):
        if self._group_name_cache:
            return self._group_name_cache
        results = self._list_rule_groups()
        if not results:
            return dict()
        group_cache = {r.get('Name', None): r.get('Arn', None) for r in results}
        self._group_name_cache = group_cache
        return group_cache

    @property
    def _stateful_rule_order(self):
        engine_options = self._get_resource_value('StatefulEngineOptions', None)
        if not engine_options:
            return 'DEFAULT_ACTION_ORDER'
        return engine_options.get('RuleOrder', 'DEFAULT_ACTION_ORDER')

    def _canonicalize_rule_group(self, name, group_type):
        """Iterates through a mixed list of ARNs and Names converting them to
        ARNs.  Also checks that the group type matches the provided group_type.
        """
        arn = None
        if ':' in name:
            arn = name
        else:
            arn = self._rule_group_name_cache.get(name, None)
            if not arn:
                self.module.fail_json('Unable to fetch ARN for rule group', name=name, group_name_cache=self._rule_group_name_cache)
        arn_info = parse_aws_arn(arn)
        if not arn_info:
            self.module.fail_json('Unable to parse ARN for rule group', arn=arn, arn_info=arn_info)
        arn_type = arn_info['resource'].split('/')[0]
        if arn_type != group_type:
            self.module.fail_json('Rule group not of expected type', name=name, arn=arn, expected_type=group_type, found_type=arn_type)
        return arn

    def _format_rulegroup_references(self, groups, strict_order):
        formated_groups = list()
        for idx, arn in enumerate(groups):
            entry = dict(ResourceArn=arn)
            if strict_order:
                entry['Priority'] = idx + 1
            formated_groups.append(entry)
        return formated_groups

    def _rulegroup_references_list(self, groups):
        return [g.get('ResourceArn') for g in groups]

    def _sorted_rulegroup_references_list(self, groups):
        sorted_list = sorted(groups, key=lambda g: g.get('Priority', None))
        return self._rulegroup_references_list(sorted_list)

    def _compare_rulegroup_references(self, current_groups, desired_groups, strict_order):
        if current_groups is None:
            return False
        if strict_order:
            current_groups = self._sorted_rulegroup_references_list(current_groups)
            return current_groups == desired_groups
        else:
            current_groups = self._rulegroup_references_list(current_groups)
            return set(current_groups) == set(desired_groups)

    def _set_engine_option(self, option_name, description, value, immutable=False, default_value=None):
        if value is None:
            return False
        engine_options = deepcopy(self._get_resource_value('StatefulEngineOptions', dict()))
        if value == engine_options.get(option_name, default_value):
            return False
        if immutable and self.original_resource:
            self.module.fail_json(msg=f'{description} can not be updated after creation')
        engine_options[option_name] = value
        return self._set_resource_value('StatefulEngineOptions', engine_options)

    def set_stateful_rule_order(self, order):
        RULE_ORDER_MAP = {'default': 'DEFAULT_ACTION_ORDER', 'strict': 'STRICT_ORDER'}
        value = RULE_ORDER_MAP.get(order)
        changed = self._set_engine_option('RuleOrder', 'Rule order', value, True, 'DEFAULT_ACTION_ORDER')
        self.changed |= changed
        return changed

    def _set_rule_groups(self, groups, group_type, parameter_name, strict_order):
        if groups is None:
            return False
        group_arns = [self._canonicalize_rule_group(g, group_type) for g in groups]
        current_groups = self._get_resource_value(parameter_name)
        if self._compare_rulegroup_references(current_groups, group_arns, strict_order):
            return False
        formated_groups = self._format_rulegroup_references(group_arns, strict_order)
        return self._set_resource_value(parameter_name, formated_groups)

    def set_stateful_rule_groups(self, groups):
        strict_order = self._stateful_rule_order == 'STRICT_ORDER'
        return self._set_rule_groups(groups, 'stateful-rulegroup', 'StatefulRuleGroupReferences', strict_order)

    def set_stateless_rule_groups(self, groups):
        return self._set_rule_groups(groups, 'stateless-rulegroup', 'StatelessRuleGroupReferences', True)

    def set_default_actions(self, key, actions, valid_actions=None):
        if actions is None:
            return False
        invalid_actions = list(set(actions) - set(valid_actions or []))
        if valid_actions and invalid_actions:
            self.module.fail_json(msg=f'{key} contains invalid actions', valid_actions=valid_actions, invalid_actions=invalid_actions, actions=actions)
        return self._set_resource_value(key, actions)

    def set_stateful_default_actions(self, actions):
        if actions is None:
            return False
        if self._stateful_rule_order != 'STRICT_ORDER':
            self.module.fail_json(msg='Stateful default actions can only be set when using strict rule order')
        valid_actions = ['aws:drop_strict', 'aws:drop_established', 'aws:alert_strict', 'aws:alert_established']
        return self.set_default_actions('StatefulDefaultActions', actions, valid_actions)

    def _set_stateless_default_actions(self, key, actions):
        valid_actions = ['aws:pass', 'aws:drop', 'aws:forward_to_sfe']
        custom_actions = self._get_resource_value('StatelessCustomActions', dict())
        custom_action_names = [a['ActionName'] for a in custom_actions]
        valid_actions.extend(custom_action_names)
        return self.set_default_actions(key, actions, valid_actions)

    def set_stateless_default_actions(self, actions):
        return self._set_stateless_default_actions('StatelessDefaultActions', actions)

    def set_stateless_fragment_default_actions(self, actions):
        return self._set_stateless_default_actions('StatelessFragmentDefaultActions', actions)

    def _normalize_policy(self, policy):
        if policy is None:
            return None
        policy = self._normalize_boto3_resource(policy)
        return policy

    def _normalize_policy_metadata(self, policy_metadata):
        if policy_metadata is None:
            return None
        return self._normalize_boto3_resource(policy_metadata, add_tags=True)

    def _normalize_policy_result(self, result):
        if result is None:
            return None
        policy = self._normalize_policy(result.get('FirewallPolicy', None))
        policy_metadata = self._normalize_policy_metadata(result.get('FirewallPolicyMetadata', None))
        result = dict()
        if policy:
            result['policy'] = policy
        if policy_metadata:
            result['policy_metadata'] = policy_metadata
        return result

    def _normalize_resource(self, resource):
        return self._normalize_policy_result(resource)

    def get_policy(self, name=None, arn=None):
        id_params = self._get_id_params(name=name, arn=arn)
        result = self._get_policy(**id_params)
        if not result:
            return None
        policy = self._normalize_policy_result(result)
        return policy

    def _format_custom_action(self, action):
        formatted_action = dict(ActionName=action['name'])
        action_definition = dict()
        if 'publish_metric_dimension_value' in action:
            values = _string_list(action['publish_metric_dimension_value'])
            dimensions = [dict(Value=v) for v in values]
            action_definition['PublishMetricAction'] = dict(Dimensions=dimensions)
        if action_definition:
            formatted_action['ActionDefinition'] = action_definition
        return formatted_action

    def _custom_action_map(self, actions):
        return {a['ActionName']: a['ActionDefinition'] for a in actions}

    def set_custom_stateless_actions(self, actions, purge_actions):
        if actions is None:
            return False
        new_action_list = [self._format_custom_action(a) for a in actions]
        new_action_map = self._custom_action_map(new_action_list)
        existing_action_map = self._custom_action_map(self._get_resource_value('StatelessCustomActions', []))
        if purge_actions:
            desired_action_map = dict()
        else:
            desired_action_map = deepcopy(existing_action_map)
        desired_action_map.update(new_action_map)
        if desired_action_map == existing_action_map:
            return False
        action_list = [dict(ActionName=k, ActionDefinition=v) for k, v in desired_action_map.items()]
        self._set_resource_value('StatelessCustomActions', action_list)

    def set_description(self, description):
        return self._set_metadata_value('Description', description)

    def _do_create_resource(self):
        metadata, resource = self._merge_changes(filter_metadata=False)
        params = metadata
        params.update(self._get_id_params())
        params['FirewallPolicy'] = resource
        response = self._create_policy(**params)
        return bool(response)

    def _generate_updated_resource(self):
        metadata, resource = self._merge_changes(filter_metadata=False)
        metadata.update(self._get_id_params())
        updated_resource = dict(FirewallPolicy=resource, FirewallPolicyMetadata=metadata)
        return updated_resource

    def _flush_create(self):
        if self._get_resource_value('StatelessDefaultActions', None) is None:
            self._set_resource_value('StatelessDefaultActions', ['aws:forward_to_sfe'])
        if self._get_resource_value('StatelessFragmentDefaultActions', None) is None:
            self._set_resource_value('StatelessFragmentDefaultActions', ['aws:forward_to_sfe'])
        return super(NetworkFirewallPolicyManager, self)._flush_create()

    def _do_update_resource(self):
        filtered_metadata_updates = self._filter_immutable_metadata_attributes(self._metadata_updates)
        filtered_resource_updates = self._resource_updates
        if not filtered_resource_updates and (not filtered_metadata_updates):
            return False
        metadata, resource = self._merge_changes()
        params = metadata
        params.update(self._get_id_params())
        params['FirewallPolicy'] = resource
        if not self.module.check_mode:
            response = self._update_policy(**params)
        return True

    def _flush_update(self):
        changed = False
        changed |= self._flush_tagging()
        changed |= super(NetworkFirewallPolicyManager, self)._flush_update()
        return changed

    def _get_policy(self, **params):
        result = self._describe_policy(**params)
        if not result:
            return None
        policy = result.get('FirewallPolicy', None)
        if policy is None:
            policy = dict()
        metadata = result.get('FirewallPolicyMetadata', None)
        self._preupdate_resource = deepcopy(policy)
        self._preupdate_metadata = deepcopy(metadata)
        return dict(FirewallPolicy=policy, FirewallPolicyMetadata=metadata)

    def get_resource(self):
        return self.get_policy()

    def _do_creation_wait(self, **params):
        all_params = self._get_id_params()
        all_params.update(params)
        return self._wait_policy_active(**all_params)

    def _do_deletion_wait(self, **params):
        all_params = self._get_id_params()
        all_params.update(params)
        return self._wait_policy_deleted(**all_params)