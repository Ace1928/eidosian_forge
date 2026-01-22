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
def _ensure_immutable_node(node):
    if node is None or node.is_immutable():
        return node
    return dns.node.ImmutableNode(node)