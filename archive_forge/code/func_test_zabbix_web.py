import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbix_web(host):
    zabbix_web = host.file('/etc/zabbix/web/zabbix.conf.php')
    ansible_variables = host.ansible.get_variables()
    zabbix_websrv = str(ansible_variables['zabbix_web_http_server'])
    if host.system_info.distribution in ['debian', 'ubuntu']:
        assert zabbix_web.user == 'www-data'
        assert zabbix_web.group == 'www-data'
    elif host.system_info.distribution == 'centos':
        if zabbix_websrv == 'apache':
            assert zabbix_web.user == 'apache'
            assert zabbix_web.group == 'apache'
        elif zabbix_websrv == 'nginx':
            assert zabbix_web.user == 'nginx'
            assert zabbix_web.group == 'nginx'
    assert zabbix_web.mode == 420