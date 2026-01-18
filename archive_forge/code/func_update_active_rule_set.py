from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_active_rule_set(client, module, name, desired_active):
    check_mode = module.check_mode
    active = ruleset_active(client, module, name)
    changed = False
    if desired_active is not None:
        if desired_active and (not active):
            if not check_mode:
                try:
                    client.set_active_receipt_rule_set(RuleSetName=name, aws_retry=True)
                except (BotoCoreError, ClientError) as e:
                    module.fail_json_aws(e, msg=f"Couldn't set active rule set to {name}.")
            changed = True
            active = True
        elif not desired_active and active:
            if not check_mode:
                deactivate_rule_set(client, module)
            changed = True
            active = False
    return (changed, active)