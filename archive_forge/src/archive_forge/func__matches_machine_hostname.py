import ipaddress
import random
import re
import socket
import time
import weakref
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union
def _matches_machine_hostname(host: str) -> bool:
    """Indicate whether ``host`` matches the hostname of this machine.

    This function compares ``host`` to the hostname as well as to the IP
    addresses of this machine. Note that it may return a false negative if this
    machine has CNAME records beyond its FQDN or IP addresses assigned to
    secondary NICs.
    """
    if host == 'localhost':
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None
    if addr and addr.is_loopback:
        return True
    try:
        host_addr_list = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME)
    except (ValueError, socket.gaierror) as _:
        host_addr_list = []
    host_ip_list = [host_addr_info[4][0] for host_addr_info in host_addr_list]
    this_host = socket.gethostname()
    if host == this_host:
        return True
    addr_list = socket.getaddrinfo(this_host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME)
    for addr_info in addr_list:
        if addr_info[3] and addr_info[3] == host:
            return True
        if addr and addr_info[4][0] == str(addr):
            return True
        if addr_info[4][0] in host_ip_list:
            return True
    return False