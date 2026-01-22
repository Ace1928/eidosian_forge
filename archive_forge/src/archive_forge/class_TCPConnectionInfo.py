from __future__ import absolute_import, division, print_function
import binascii
import contextlib
import datetime
import errno
import math
import mmap
import os
import re
import select
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.compat.datetime import utcnow
class TCPConnectionInfo(object):
    """
    This is a generic TCP Connection Info strategy class that relies
    on the psutil module, which is not ideal for targets, but necessary
    for cross platform support.

    A subclass may wish to override some or all of these methods.
      - _get_exclude_ips()
      - get_active_connections()

    All subclasses MUST define platform and distribution (which may be None).
    """
    platform = 'Generic'
    distribution = None
    match_all_ips = {socket.AF_INET: '0.0.0.0', socket.AF_INET6: '::'}
    ipv4_mapped_ipv6_address = {'prefix': '::ffff', 'match_all': '::ffff:0.0.0.0'}

    def __new__(cls, *args, **kwargs):
        new_cls = get_platform_subclass(TCPConnectionInfo)
        return super(cls, new_cls).__new__(new_cls)

    def __init__(self, module):
        self.module = module
        self.ips = _convert_host_to_ip(module.params['host'])
        self.port = int(self.module.params['port'])
        self.exclude_ips = self._get_exclude_ips()
        if not HAS_PSUTIL:
            module.fail_json(msg=missing_required_lib('psutil'), exception=PSUTIL_IMP_ERR)

    def _get_exclude_ips(self):
        exclude_hosts = self.module.params['exclude_hosts']
        exclude_ips = []
        if exclude_hosts is not None:
            for host in exclude_hosts:
                exclude_ips.extend(_convert_host_to_ip(host))
        return exclude_ips

    def get_active_connections_count(self):
        active_connections = 0
        for p in psutil.process_iter():
            try:
                if hasattr(p, 'get_connections'):
                    connections = p.get_connections(kind='inet')
                else:
                    connections = p.connections(kind='inet')
            except psutil.Error:
                continue
            for conn in connections:
                if conn.status not in self.module.params['active_connection_states']:
                    continue
                if hasattr(conn, 'local_address'):
                    local_ip, local_port = conn.local_address
                else:
                    local_ip, local_port = conn.laddr
                if self.port != local_port:
                    continue
                if hasattr(conn, 'remote_address'):
                    remote_ip, remote_port = conn.remote_address
                else:
                    remote_ip, remote_port = conn.raddr
                if (conn.family, remote_ip) in self.exclude_ips:
                    continue
                if any(((conn.family, local_ip) in self.ips, (conn.family, self.match_all_ips[conn.family]) in self.ips, local_ip.startswith(self.ipv4_mapped_ipv6_address['prefix']) and (conn.family, self.ipv4_mapped_ipv6_address['match_all']) in self.ips)):
                    active_connections += 1
        return active_connections