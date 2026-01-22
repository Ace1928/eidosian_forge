import base64
import contextlib
import functools
import hashlib
import struct
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast
import dns._features
import dns.exception
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.transaction
import dns.zone
from dns.dnssectypes import Algorithm, DSDigest, NSEC3Hash
from dns.exception import (  # pylint: disable=W0611
from dns.rdtypes.ANY.CDNSKEY import CDNSKEY
from dns.rdtypes.ANY.CDS import CDS
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.ANY.DS import DS
from dns.rdtypes.ANY.NSEC import NSEC, Bitmap
from dns.rdtypes.ANY.NSEC3PARAM import NSEC3PARAM
from dns.rdtypes.ANY.RRSIG import RRSIG, sigtime_to_posixtime
from dns.rdtypes.dnskeybase import Flag
def _sign_zone_nsec(zone: dns.zone.Zone, txn: dns.transaction.Transaction, rrset_signer: Optional[RRsetSigner]=None) -> None:
    """NSEC zone signer"""

    def _txn_add_nsec(txn: dns.transaction.Transaction, name: dns.name.Name, next_secure: Optional[dns.name.Name], rdclass: dns.rdataclass.RdataClass, ttl: int, rrset_signer: Optional[RRsetSigner]=None) -> None:
        """NSEC zone signer helper"""
        mandatory_types = set([dns.rdatatype.RdataType.RRSIG, dns.rdatatype.RdataType.NSEC])
        node = txn.get_node(name)
        if node and next_secure:
            types = set([rdataset.rdtype for rdataset in node.rdatasets]) | mandatory_types
            windows = Bitmap.from_rdtypes(list(types))
            rrset = dns.rrset.from_rdata(name, ttl, NSEC(rdclass=rdclass, rdtype=dns.rdatatype.RdataType.NSEC, next=next_secure, windows=windows))
            txn.add(rrset)
            if rrset_signer:
                rrset_signer(txn, rrset)
    rrsig_ttl = zone.get_soa().minimum
    delegation = None
    last_secure = None
    for name in sorted(txn.iterate_names()):
        if delegation and name.is_subdomain(delegation):
            continue
        elif txn.get(name, dns.rdatatype.NS) and name != zone.origin:
            delegation = name
        else:
            delegation = None
        if rrset_signer:
            node = txn.get_node(name)
            if node:
                for rdataset in node.rdatasets:
                    if rdataset.rdtype == dns.rdatatype.RRSIG:
                        continue
                    elif delegation and rdataset.rdtype != dns.rdatatype.DS:
                        continue
                    else:
                        rrset = dns.rrset.from_rdata(name, rdataset.ttl, *rdataset)
                        rrset_signer(txn, rrset)
        if last_secure is not None:
            _txn_add_nsec(txn, last_secure, name, zone.rdclass, rrsig_ttl, rrset_signer)
        last_secure = name
    if last_secure:
        _txn_add_nsec(txn, last_secure, zone.origin, zone.rdclass, rrsig_ttl, rrset_signer)