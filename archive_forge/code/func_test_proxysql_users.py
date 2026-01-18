import os
import pytest
import testinfra.utils.ansible_runner
@pytest.mark.parametrize(proxysql_user_attributes, [('proxysql', 'proxysql')])
def test_proxysql_users(host, user_name, group_name):
    u = host.user(user_name)
    assert u.exists
    assert u.group == group_name