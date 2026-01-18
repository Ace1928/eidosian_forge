import os
import testinfra.utils.ansible_runner
def test_thp_service_file(host):
    f = host.file('/etc/systemd/system/disable-transparent-huge-pages.service')
    assert f.exists
    assert f.user == 'root'
    assert f.group == 'root'