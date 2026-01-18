import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbixproxy_running_and_enabled(host):
    zabbix = host.service('zabbix-proxy')
    if host.system_info.distribution == 'centos':
        assert zabbix.is_enabled
        assert zabbix.is_running
    else:
        assert zabbix.is_running