import os
import pytest
import testinfra.utils.ansible_runner
@pytest.mark.parametrize('proxysql_service', ['proxysql'])
def test_proxysql_services(host, proxysql_service):
    svc = host.service(proxysql_service)
    assert svc.is_enabled
    assert svc.is_running