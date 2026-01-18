import os
import pytest
import testinfra.utils.ansible_runner
def test_zabbix_agent_dot_conf(zabbix_agent_conf):
    assert zabbix_agent_conf.contains('TLSAccept=psk')
    assert zabbix_agent_conf.contains('TLSPSKIdentity=my_Identity')
    assert zabbix_agent_conf.contains('TLSPSKFile=/data/certs/zabbix.psk')