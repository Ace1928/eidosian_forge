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
class LRUCacheNode(object):
    """LRUCache node."""

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = self
        self.next = self

    def link_before(self, node):
        self.prev = node.prev
        self.next = node
        node.prev.next = self
        node.prev = self

    def link_after(self, node):
        self.prev = node
        self.next = node.next
        node.next.prev = self
        node.next = self

    def unlink(self):
        self.next.prev = self.prev
        self.prev.next = self.next