import collections
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.serial
import dns.ttl
def _raise_if_not_empty(self, method, args):
    if len(args) != 0:
        raise TypeError(f'extra parameters to {method}')