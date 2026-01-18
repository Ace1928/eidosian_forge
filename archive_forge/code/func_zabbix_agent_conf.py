import os
import pytest
import testinfra.utils.ansible_runner
@pytest.fixture
def zabbix_agent_conf(host):
    if host.system_info.distribution in ['opensuse']:
        passwd = host.file('/etc/zabbix/zabbix-agentd.conf')
    else:
        passwd = host.file('/etc/zabbix/zabbix_agent2.conf')
    return passwd