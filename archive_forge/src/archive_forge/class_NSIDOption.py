import binascii
import math
import socket
import struct
from typing import Any, Dict, Optional, Union
import dns.enum
import dns.inet
import dns.rdata
import dns.wire
class NSIDOption(Option):

    def __init__(self, nsid: bytes):
        super().__init__(OptionType.NSID)
        self.nsid = nsid

    def to_wire(self, file: Any=None) -> Optional[bytes]:
        if file:
            file.write(self.nsid)
            return None
        else:
            return self.nsid

    def to_text(self) -> str:
        if all((c >= 32 and c <= 126 for c in self.nsid)):
            value = self.nsid.decode()
        else:
            value = binascii.hexlify(self.nsid).decode()
        return f'NSID {value}'

    @classmethod
    def from_wire_parser(cls, otype: Union[OptionType, str], parser: dns.wire.Parser) -> Option:
        return cls(parser.get_remaining())