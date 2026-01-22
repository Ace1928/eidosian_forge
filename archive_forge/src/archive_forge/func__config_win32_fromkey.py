import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def _config_win32_fromkey(self, key, always_try_domain):
    try:
        servers, rtype = _winreg.QueryValueEx(key, 'NameServer')
    except WindowsError:
        servers = None
    if servers:
        self._config_win32_nameservers(servers)
    if servers or always_try_domain:
        try:
            dom, rtype = _winreg.QueryValueEx(key, 'Domain')
            if dom:
                self._config_win32_domain(dom)
        except WindowsError:
            pass
    else:
        try:
            servers, rtype = _winreg.QueryValueEx(key, 'DhcpNameServer')
        except WindowsError:
            servers = None
        if servers:
            self._config_win32_nameservers(servers)
            try:
                dom, rtype = _winreg.QueryValueEx(key, 'DhcpDomain')
                if dom:
                    self._config_win32_domain(dom)
            except WindowsError:
                pass
    try:
        search, rtype = _winreg.QueryValueEx(key, 'SearchList')
    except WindowsError:
        search = None
    if search:
        self._config_win32_search(search)