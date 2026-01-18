import os
import pytest
from pathlib import Path
import testinfra.utils.ansible_runner
def test_zabbix_server_dot_conf(host):
    zabbix_server_conf = host.file('/etc/zabbix/zabbix_server.conf')
    assert zabbix_server_conf.exists
    assert zabbix_server_conf.user == 'zabbix'
    assert zabbix_server_conf.group == 'zabbix'
    assert zabbix_server_conf.mode == 416
    assert zabbix_server_conf.contains('ListenPort=10051')
    assert zabbix_server_conf.contains('DebugLevel=3')