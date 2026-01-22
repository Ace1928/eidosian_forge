import contextlib
import random
import socket
import sys
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
import dns._ddr
import dns.edns
import dns.exception
import dns.flags
import dns.inet
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.nameserver
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.svcbbase
import dns.reversename
import dns.tsig
class _Resolution:
    """Helper class for dns.resolver.Resolver.resolve().

    All of the "business logic" of resolution is encapsulated in this
    class, allowing us to have multiple resolve() implementations
    using different I/O schemes without copying all of the
    complicated logic.

    This class is a "friend" to dns.resolver.Resolver and manipulates
    resolver data structures directly.
    """

    def __init__(self, resolver: 'BaseResolver', qname: Union[dns.name.Name, str], rdtype: Union[dns.rdatatype.RdataType, str], rdclass: Union[dns.rdataclass.RdataClass, str], tcp: bool, raise_on_no_answer: bool, search: Optional[bool]) -> None:
        if isinstance(qname, str):
            qname = dns.name.from_text(qname, None)
        rdtype = dns.rdatatype.RdataType.make(rdtype)
        if dns.rdatatype.is_metatype(rdtype):
            raise NoMetaqueries
        rdclass = dns.rdataclass.RdataClass.make(rdclass)
        if dns.rdataclass.is_metaclass(rdclass):
            raise NoMetaqueries
        self.resolver = resolver
        self.qnames_to_try = resolver._get_qnames_to_try(qname, search)
        self.qnames = self.qnames_to_try[:]
        self.rdtype = rdtype
        self.rdclass = rdclass
        self.tcp = tcp
        self.raise_on_no_answer = raise_on_no_answer
        self.nxdomain_responses: Dict[dns.name.Name, dns.message.QueryMessage] = {}
        self.qname = dns.name.empty
        self.nameservers: List[dns.nameserver.Nameserver] = []
        self.current_nameservers: List[dns.nameserver.Nameserver] = []
        self.errors: List[ErrorTuple] = []
        self.nameserver: Optional[dns.nameserver.Nameserver] = None
        self.tcp_attempt = False
        self.retry_with_tcp = False
        self.request: Optional[dns.message.QueryMessage] = None
        self.backoff = 0.0

    def next_request(self) -> Tuple[Optional[dns.message.QueryMessage], Optional[Answer]]:
        """Get the next request to send, and check the cache.

        Returns a (request, answer) tuple.  At most one of request or
        answer will not be None.
        """
        while len(self.qnames) > 0:
            self.qname = self.qnames.pop(0)
            if self.resolver.cache:
                answer = self.resolver.cache.get((self.qname, self.rdtype, self.rdclass))
                if answer is not None:
                    if answer.rrset is None and self.raise_on_no_answer:
                        raise NoAnswer(response=answer.response)
                    else:
                        return (None, answer)
                answer = self.resolver.cache.get((self.qname, dns.rdatatype.ANY, self.rdclass))
                if answer is not None and answer.response.rcode() == dns.rcode.NXDOMAIN:
                    self.nxdomain_responses[self.qname] = answer.response
                    continue
            request = dns.message.make_query(self.qname, self.rdtype, self.rdclass)
            if self.resolver.keyname is not None:
                request.use_tsig(self.resolver.keyring, self.resolver.keyname, algorithm=self.resolver.keyalgorithm)
            request.use_edns(self.resolver.edns, self.resolver.ednsflags, self.resolver.payload, options=self.resolver.ednsoptions)
            if self.resolver.flags is not None:
                request.flags = self.resolver.flags
            self.nameservers = self.resolver._enrich_nameservers(self.resolver._nameservers, self.resolver.nameserver_ports, self.resolver.port)
            if self.resolver.rotate:
                random.shuffle(self.nameservers)
            self.current_nameservers = self.nameservers[:]
            self.errors = []
            self.nameserver = None
            self.tcp_attempt = False
            self.retry_with_tcp = False
            self.request = request
            self.backoff = 0.1
            return (request, None)
        raise NXDOMAIN(qnames=self.qnames_to_try, responses=self.nxdomain_responses)

    def next_nameserver(self) -> Tuple[dns.nameserver.Nameserver, bool, float]:
        if self.retry_with_tcp:
            assert self.nameserver is not None
            assert not self.nameserver.is_always_max_size()
            self.tcp_attempt = True
            self.retry_with_tcp = False
            return (self.nameserver, True, 0)
        backoff = 0.0
        if not self.current_nameservers:
            if len(self.nameservers) == 0:
                raise NoNameservers(request=self.request, errors=self.errors)
            self.current_nameservers = self.nameservers[:]
            backoff = self.backoff
            self.backoff = min(self.backoff * 2, 2)
        self.nameserver = self.current_nameservers.pop(0)
        self.tcp_attempt = self.tcp or self.nameserver.is_always_max_size()
        return (self.nameserver, self.tcp_attempt, backoff)

    def query_result(self, response: Optional[dns.message.Message], ex: Optional[Exception]) -> Tuple[Optional[Answer], bool]:
        assert self.nameserver is not None
        if ex:
            assert response is None
            self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), ex, response))
            if isinstance(ex, dns.exception.FormError) or isinstance(ex, EOFError) or isinstance(ex, OSError) or isinstance(ex, NotImplementedError):
                self.nameservers.remove(self.nameserver)
            elif isinstance(ex, dns.message.Truncated):
                if self.tcp_attempt:
                    self.nameservers.remove(self.nameserver)
                else:
                    self.retry_with_tcp = True
            return (None, False)
        assert response is not None
        assert isinstance(response, dns.message.QueryMessage)
        rcode = response.rcode()
        if rcode == dns.rcode.NOERROR:
            try:
                answer = Answer(self.qname, self.rdtype, self.rdclass, response, self.nameserver.answer_nameserver(), self.nameserver.answer_port())
            except Exception as e:
                self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), e, response))
                self.nameservers.remove(self.nameserver)
                return (None, False)
            if self.resolver.cache:
                self.resolver.cache.put((self.qname, self.rdtype, self.rdclass), answer)
            if answer.rrset is None and self.raise_on_no_answer:
                raise NoAnswer(response=answer.response)
            return (answer, True)
        elif rcode == dns.rcode.NXDOMAIN:
            try:
                answer = Answer(self.qname, dns.rdatatype.ANY, dns.rdataclass.IN, response)
            except Exception as e:
                self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), e, response))
                self.nameservers.remove(self.nameserver)
                return (None, False)
            self.nxdomain_responses[self.qname] = response
            if self.resolver.cache:
                self.resolver.cache.put((self.qname, dns.rdatatype.ANY, self.rdclass), answer)
            return (None, True)
        elif rcode == dns.rcode.YXDOMAIN:
            yex = YXDOMAIN()
            self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), yex, response))
            raise yex
        else:
            if rcode != dns.rcode.SERVFAIL or not self.resolver.retry_servfail:
                self.nameservers.remove(self.nameserver)
            self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), dns.rcode.to_text(rcode), response))
            return (None, False)