from typing import Any, List, Optional, Union
import dns.message
import dns.name
import dns.opcode
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.tsig
class UpdateSection(dns.enum.IntEnum):
    """Update sections"""
    ZONE = 0
    PREREQ = 1
    UPDATE = 2
    ADDITIONAL = 3

    @classmethod
    def _maximum(cls):
        return 3