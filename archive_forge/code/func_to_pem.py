from abc import ABC, abstractmethod  # pylint: disable=no-name-in-module
from typing import Any, Optional, Type
import dns.rdataclass
import dns.rdatatype
from dns.dnssectypes import Algorithm
from dns.exception import AlgorithmKeyMismatch
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.dnskeybase import Flag
@abstractmethod
def to_pem(self, password: Optional[bytes]=None) -> bytes:
    """Return private key as PEM-encoded PKCS#8"""