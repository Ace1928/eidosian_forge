from __future__ import (absolute_import, division, print_function)
import os
import testinfra.utils.ansible_runner
def test_packages(host):
    p = host.package('grafana')
    assert p.is_installed