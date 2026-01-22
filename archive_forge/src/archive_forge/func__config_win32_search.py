import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def _config_win32_search(self, search):
    search = str(search)
    split_char = self._determine_split_char(search)
    search_list = search.split(split_char)
    for s in search_list:
        if s not in self.search:
            self.search.append(dns.name.from_text(s))