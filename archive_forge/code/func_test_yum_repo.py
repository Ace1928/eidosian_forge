from __future__ import (absolute_import, division, print_function)
import os
import testinfra.utils.ansible_runner
def test_yum_repo(host):
    if host.system_info.distribution in ['centos', 'redhat', 'fedora']:
        f = host.file('/etc/yum.repos.d/grafana.repo')
        assert f.exists