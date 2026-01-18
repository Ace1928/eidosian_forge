import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_or_append_rule(new_rule, existing_rule, purge_transitions, lifecycle_obj):
    changed = False
    if existing_rule['Status'] != new_rule['Status']:
        if not new_rule.get('Transitions') and existing_rule.get('Transitions'):
            new_rule['Transitions'] = existing_rule['Transitions']
        if not new_rule.get('Expiration') and existing_rule.get('Expiration'):
            new_rule['Expiration'] = existing_rule['Expiration']
        if not new_rule.get('NoncurrentVersionExpiration') and existing_rule.get('NoncurrentVersionExpiration'):
            new_rule['NoncurrentVersionExpiration'] = existing_rule['NoncurrentVersionExpiration']
        lifecycle_obj['Rules'].append(new_rule)
        changed = True
        appended = True
    else:
        if not purge_transitions:
            merge_transitions(new_rule, existing_rule)
        if compare_rule(new_rule, existing_rule, purge_transitions):
            lifecycle_obj['Rules'].append(new_rule)
            appended = True
        else:
            lifecycle_obj['Rules'].append(new_rule)
            changed = True
            appended = True
    return (changed, appended)