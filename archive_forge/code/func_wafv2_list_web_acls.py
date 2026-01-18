from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def wafv2_list_web_acls(wafv2, scope, fail_json_aws, nextmarker=None):
    req_obj = {'Scope': scope, 'Limit': 100}
    if nextmarker:
        req_obj['NextMarker'] = nextmarker
    try:
        response = wafv2.list_web_acls(**req_obj)
    except (BotoCoreError, ClientError) as e:
        fail_json_aws(e, msg='Failed to list wafv2 web acl')
    if response.get('NextMarker'):
        response['WebACLs'] += wafv2_list_web_acls(wafv2, scope, fail_json_aws, nextmarker=response.get('NextMarker')).get('WebACLs')
    return response