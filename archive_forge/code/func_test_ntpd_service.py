import os
import testinfra.utils.ansible_runner
def test_ntpd_service(host):
    ntpd = host.service('ntpd')
    if ntpd.is_running:
        assert ntpd.is_enabled
    else:
        ntp = host.service('ntp')
        if ntp.is_running:
            assert ntp.is_enabled
        else:
            chronyd = host.service('chronyd')
            assert chronyd.is_running
            assert chronyd.is_enabled