import re
import sys
from typing import Any, Iterable, List, Optional, Set, Tuple, Union
import dns.exception
import dns.grange
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
def _calculate_index(counter: int, offset_sign: str, offset: int) -> int:
    """Calculate the index from the counter and offset."""
    if offset_sign == '-':
        offset *= -1
    return counter + offset