from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native

    input: ip_address and netmask in dot notation for IPv4, expanded netmask is not supported for IPv6
           netmask as int or a str representaiton of int is also accepted
    output: netmask length as int
    