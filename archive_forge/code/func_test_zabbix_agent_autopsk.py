import pytest
import os
import testinfra.utils.ansible_runner
def test_zabbix_agent_autopsk(host):
    psk_file = host.file('/etc/zabbix/tls_psk_auto.secret')
    assert psk_file.user == 'zabbix'
    assert psk_file.group == 'zabbix'
    assert psk_file.mode == 256
    assert psk_file.size == 64