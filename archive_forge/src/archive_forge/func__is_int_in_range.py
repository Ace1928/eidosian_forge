import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def _is_int_in_range(value, start, end):
    """Try to convert value to int and check if it lies within
    range 'start' to 'end'.

    :param value: value to verify
    :param start: start number of range
    :param end: end number of range
    :returns: bool
    """
    try:
        val = int(value)
    except (ValueError, TypeError):
        return False
    return start <= val <= end