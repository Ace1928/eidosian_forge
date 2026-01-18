import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_sqs_queue(module, client, queue_url):
    check_mode = module.check_mode
    changed = False
    existing_attributes = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['All'], aws_retry=True)['Attributes']
    new_attributes = snake_dict_to_camel_dict(module.params, capitalize_first=True)
    attributes_to_set = dict()
    if module.params.get('policy') is not None:
        policy = module.params.get('policy')
        current_value = existing_attributes.get('Policy', '{}')
        current_policy = json.loads(current_value)
        if compare_policies(current_policy, policy):
            attributes_to_set['Policy'] = json.dumps(policy)
            changed = True
    if module.params.get('redrive_policy') is not None:
        policy = module.params.get('redrive_policy')
        current_value = existing_attributes.get('RedrivePolicy', '{}')
        current_policy = json.loads(current_value)
        if compare_policies(current_policy, policy):
            attributes_to_set['RedrivePolicy'] = json.dumps(policy)
            changed = True
    for attribute, value in existing_attributes.items():
        if attribute in ['Policy', 'RedrivePolicy']:
            continue
        if attribute not in new_attributes.keys():
            continue
        if new_attributes.get(attribute) is None:
            continue
        new_value = new_attributes[attribute]
        if isinstance(new_value, bool):
            new_value = str(new_value).lower()
            value = str(value).lower()
        if str(new_value) == str(value):
            continue
        attributes_to_set[attribute] = str(new_value)
        changed = True
    if changed and (not check_mode):
        client.set_queue_attributes(QueueUrl=queue_url, Attributes=attributes_to_set, aws_retry=True)
    return (changed, existing_attributes.get('queue_arn'))