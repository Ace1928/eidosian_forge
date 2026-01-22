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
def _checked_delete_rdataset(self, name, rdtype, covers):
    for check in self._check_delete_rdataset:
        check(self, name, rdtype, covers)
    self._delete_rdataset(name, rdtype, covers)