import re
import struct
import sys
import eventlet
from eventlet import patcher
from eventlet.green import _socket_nodns
from eventlet.green import os
from eventlet.green import time
from eventlet.green import select
from eventlet.green import ssl
def _getaddrinfo_lookup(host, family, flags):
    """Resolve a hostname to a list of addresses

    Helper function for getaddrinfo.
    """
    if flags & socket.AI_NUMERICHOST:
        _raise_new_error(EAI_NONAME_ERROR)
    addrs = []
    if family == socket.AF_UNSPEC:
        err = None
        for use_network in [False, True]:
            for qfamily in [socket.AF_INET6, socket.AF_INET]:
                try:
                    answer = resolve(host, qfamily, False, use_network=use_network)
                except socket.gaierror as e:
                    if e.errno not in (socket.EAI_AGAIN, EAI_NONAME_ERROR.errno, EAI_NODATA_ERROR.errno):
                        raise
                    err = e
                else:
                    if answer.rrset:
                        addrs.extend((rr.address for rr in answer.rrset))
            if addrs:
                break
        if err is not None and (not addrs):
            raise err
    elif family == socket.AF_INET6 and flags & socket.AI_V4MAPPED:
        answer = resolve(host, socket.AF_INET6, False)
        if answer.rrset:
            addrs = [rr.address for rr in answer.rrset]
        if not addrs or flags & socket.AI_ALL:
            answer = resolve(host, socket.AF_INET, False)
            if answer.rrset:
                addrs = ['::ffff:' + rr.address for rr in answer.rrset]
    else:
        answer = resolve(host, family, False)
        if answer.rrset:
            addrs = [rr.address for rr in answer.rrset]
    return (str(answer.qname), addrs)