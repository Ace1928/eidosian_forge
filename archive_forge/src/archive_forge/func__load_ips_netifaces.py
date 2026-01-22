from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _load_ips_netifaces() -> None:
    """load ip addresses with netifaces"""
    import netifaces
    global LOCALHOST
    local_ips = []
    public_ips = []
    for iface in netifaces.interfaces():
        ipv4s = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
        for entry in ipv4s:
            addr = entry.get('addr')
            if not addr:
                continue
            if not (iface.startswith('lo') or addr.startswith('127.')):
                public_ips.append(addr)
            elif not LOCALHOST:
                LOCALHOST = addr
            local_ips.append(addr)
    if not LOCALHOST:
        LOCALHOST = '127.0.0.1'
        local_ips.insert(0, LOCALHOST)
    local_ips.extend(['0.0.0.0', ''])
    LOCAL_IPS[:] = _uniq_stable(local_ips)
    PUBLIC_IPS[:] = _uniq_stable(public_ips)