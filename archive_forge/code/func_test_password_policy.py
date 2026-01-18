import boto
import time
from tests.compat import unittest
def test_password_policy(self):
    iam = boto.connect_iam()
    try:
        initial_policy_result = iam.get_account_password_policy()
    except boto.exception.BotoServerError as srv_error:
        initial_policy = None
        if srv_error.status != 404:
            raise srv_error
    test_min_length = 88
    iam.update_account_password_policy(minimum_password_length=test_min_length)
    new_policy = iam.get_account_password_policy()
    new_min_length = new_policy['get_account_password_policy_response']['get_account_password_policy_result']['password_policy']['minimum_password_length']
    if test_min_length != int(new_min_length):
        raise Exception('Failed to update account password policy')
    test_policy = ''
    iam.delete_account_password_policy()
    try:
        test_policy = iam.get_account_password_policy()
    except boto.exception.BotoServerError as srv_error:
        test_policy = None
        if srv_error.status != 404:
            raise srv_error
    if test_policy is not None:
        raise Exception('Failed to delete account password policy')
    if initial_policy:
        p = initial_policy['get_account_password_policy_response']['get_account_password_policy_result']['password_policy']
        iam.update_account_password_policy(minimum_password_length=int(p['minimum_password_length']), allow_users_to_change_password=bool(p['allow_users_to_change_password']), hard_expiry=bool(p['hard_expiry']), max_password_age=int(p['max_password_age']), password_reuse_prevention=int(p['password_reuse_prevention']), require_lowercase_characters=bool(p['require_lowercase_characters']), require_numbers=bool(p['require_numbers']), require_symbols=bool(p['require_symbols']), require_uppercase_characters=bool(p['require_uppercase_characters']))