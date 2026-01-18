import boto
import time
import json
from tests.compat import unittest
def test_policy_actions(self):
    iam = boto.connect_iam()
    time_suffix = time.time()
    rolename = 'boto-test-role-%d' % time_suffix
    groupname = 'boto-test-group-%d' % time_suffix
    username = 'boto-test-user-%d' % time_suffix
    policyname = 'TestPolicyName-%d' % time_suffix
    iam.create_role(rolename)
    iam.create_group(groupname)
    iam.create_user(username)
    policy_doc = {'Version': '2012-10-17', 'Id': 'TestPermission', 'Statement': [{'Sid': 'TestSid', 'Action': 's3:*', 'Effect': 'Deny', 'Resource': 'arn:aws:s3:::*'}]}
    policy_json = json.dumps(policy_doc)
    policy = iam.create_policy(policyname, policy_json)
    policy_copy = iam.get_policy(policy.arn)
    if not policy_copy.arn == policy.arn:
        raise Exception('Policies not equal.')
    result = iam.list_entities_for_policy(policy.arn)['list_entities_for_policy_response']['list_entities_for_policy_result']
    if not len(result['policy_roles']) == 0:
        raise Exception('Roles when not expected')
    if not len(result['policy_groups']) == 0:
        raise Exception('Groups when not expected')
    if not len(result['policy_users']) == 0:
        raise Exception('Users when not expected')
    iam.attach_role_policy(policy.arn, rolename)
    iam.attach_group_policy(policy.arn, groupname)
    iam.attach_user_policy(policy.arn, username)
    result = iam.list_entities_for_policy(policy.arn)['list_entities_for_policy_response']['list_entities_for_policy_result']
    if not len(result['policy_roles']) == 1:
        raise Exception('Roles expected')
    if not len(result['policy_groups']) == 1:
        raise Exception('Groups expected')
    if not len(result['policy_users']) == 1:
        raise Exception('Users expected')
    iam.detach_role_policy(policy.arn, rolename)
    iam.detach_group_policy(policy.arn, groupname)
    iam.detach_user_policy(policy.arn, username)
    iam.delete_policy(policy.arn)
    iam.delete_role(rolename)
    iam.delete_user(username)
    iam.delete_group(groupname)