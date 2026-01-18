import os
import pytest
import testinfra.utils.ansible_runner
@pytest.mark.parametrize(proxysql_file_attributes, [('/root/.my.cnf', None, None, 384), ('/etc/proxysql.cnf', 'proxysql', 'proxysql', 420)])
def test_proxysql_files(host, proxysql_file, proxysql_file_user, proxysql_file_group, proxysql_file_mode):
    f = host.file(proxysql_file)
    assert f.exists
    assert f.is_file
    if proxysql_file_user:
        assert f.user == proxysql_file_user
    if proxysql_file_group:
        assert f.group == proxysql_file_group
    if proxysql_file_mode:
        assert f.mode == proxysql_file_mode