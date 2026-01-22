from kazoo import client
from kazoo import exceptions as k_exc
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow import exceptions as exc
from taskflow import logging
def _parse_hosts(hosts):
    if isinstance(hosts, str):
        return hosts.strip()
    if isinstance(hosts, dict):
        host_ports = []
        for k, v in hosts.items():
            host_ports.append('%s:%s' % (k, v))
        hosts = host_ports
    if isinstance(hosts, (list, set, tuple)):
        return ','.join([str(h) for h in hosts])
    return hosts