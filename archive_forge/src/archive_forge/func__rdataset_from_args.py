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
def _rdataset_from_args(self, method, deleting, args):
    try:
        arg = args.popleft()
        if isinstance(arg, dns.rrset.RRset):
            rdataset = arg.to_rdataset()
        elif isinstance(arg, dns.rdataset.Rdataset):
            rdataset = arg
        else:
            if deleting:
                ttl = 0
            else:
                if isinstance(arg, int):
                    ttl = arg
                    if ttl > dns.ttl.MAX_TTL:
                        raise ValueError(f'{method}: TTL value too big')
                else:
                    raise TypeError(f'{method}: expected a TTL')
                arg = args.popleft()
            if isinstance(arg, dns.rdata.Rdata):
                rdataset = dns.rdataset.from_rdata(ttl, arg)
            else:
                raise TypeError(f'{method}: expected an Rdata')
        return rdataset
    except IndexError:
        if deleting:
            return None
        else:
            raise TypeError(f'{method}: expected more arguments')