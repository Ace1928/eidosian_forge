from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def wafv2_list_rule_groups(wafv2, scope, fail_json_aws, nextmarker=None):
    req_obj = {'Scope': scope, 'Limit': 100}
    if nextmarker:
        req_obj['NextMarker'] = nextmarker
    try:
        response = wafv2.list_rule_groups(**req_obj)
    except (BotoCoreError, ClientError) as e:
        fail_json_aws(e, msg='Failed to list wafv2 rule group')
    if response.get('NextMarker'):
        response['RuleGroups'] += wafv2_list_rule_groups(wafv2, scope, fail_json_aws, nextmarker=response.get('NextMarker')).get('RuleGroups')
    return response