import re
import struct
import sys
import eventlet
from eventlet import patcher
from eventlet.green import _socket_nodns
from eventlet.green import os
from eventlet.green import time
from eventlet.green import select
from eventlet.green import ssl
class HostsResolver:
    """Class to parse the hosts file

    Attributes
    ----------

    :fname: The filename of the hosts file in use.
    :interval: The time between checking for hosts file modification
    """
    LINES_RE = re.compile('\n        \\s*  # Leading space\n        ([^\\r\\n#]*?)  # The actual match, non-greedy so as not to include trailing space\n        \\s*  # Trailing space\n        (?:[#][^\\r\\n]+)?  # Comments\n        (?:$|[\\r\\n]+)  # EOF or newline\n    ', re.VERBOSE)

    def __init__(self, fname=None, interval=HOSTS_TTL):
        self._v4 = {}
        self._v6 = {}
        self._aliases = {}
        self.interval = interval
        self.fname = fname
        if fname is None:
            if os.name == 'posix':
                self.fname = '/etc/hosts'
            elif os.name == 'nt':
                self.fname = os.path.expandvars('%SystemRoot%\\system32\\drivers\\etc\\hosts')
        self._last_load = 0
        if self.fname:
            self._load()

    def _readlines(self):
        """Read the contents of the hosts file

        Return list of lines, comment lines and empty lines are
        excluded.

        Note that this performs disk I/O so can be blocking.
        """
        try:
            with open(self.fname, 'rb') as fp:
                fdata = fp.read()
        except OSError:
            return []
        udata = fdata.decode(errors='ignore')
        return filter(None, self.LINES_RE.findall(udata))

    def _load(self):
        """Load hosts file

        This will unconditionally (re)load the data from the hosts
        file.
        """
        lines = self._readlines()
        self._v4.clear()
        self._v6.clear()
        self._aliases.clear()
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            ip = parts.pop(0)
            if is_ipv4_addr(ip):
                ipmap = self._v4
            elif is_ipv6_addr(ip):
                if ip.startswith('fe80'):
                    continue
                ipmap = self._v6
            else:
                continue
            cname = parts.pop(0).lower()
            ipmap[cname] = ip
            for alias in parts:
                alias = alias.lower()
                ipmap[alias] = ip
                self._aliases[alias] = cname
        self._last_load = time.time()

    def query(self, qname, rdtype=dns.rdatatype.A, rdclass=dns.rdataclass.IN, tcp=False, source=None, raise_on_no_answer=True):
        """Query the hosts file

        The known rdtypes are dns.rdatatype.A, dns.rdatatype.AAAA and
        dns.rdatatype.CNAME.

        The ``rdclass`` parameter must be dns.rdataclass.IN while the
        ``tcp`` and ``source`` parameters are ignored.

        Return a HostAnswer instance or raise a dns.resolver.NoAnswer
        exception.
        """
        now = time.time()
        if self._last_load + self.interval < now:
            self._load()
        rdclass = dns.rdataclass.IN
        if isinstance(qname, str):
            name = qname
            qname = dns.name.from_text(qname)
        elif isinstance(qname, bytes):
            name = qname.decode('ascii')
            qname = dns.name.from_text(qname)
        else:
            name = str(qname)
        name = name.lower()
        rrset = dns.rrset.RRset(qname, rdclass, rdtype)
        rrset.ttl = self._last_load + self.interval - now
        if rdclass == dns.rdataclass.IN and rdtype == dns.rdatatype.A:
            addr = self._v4.get(name)
            if not addr and qname.is_absolute():
                addr = self._v4.get(name[:-1])
            if addr:
                rrset.add(dns.rdtypes.IN.A.A(rdclass, rdtype, addr))
        elif rdclass == dns.rdataclass.IN and rdtype == dns.rdatatype.AAAA:
            addr = self._v6.get(name)
            if not addr and qname.is_absolute():
                addr = self._v6.get(name[:-1])
            if addr:
                rrset.add(dns.rdtypes.IN.AAAA.AAAA(rdclass, rdtype, addr))
        elif rdclass == dns.rdataclass.IN and rdtype == dns.rdatatype.CNAME:
            cname = self._aliases.get(name)
            if not cname and qname.is_absolute():
                cname = self._aliases.get(name[:-1])
            if cname:
                rrset.add(dns.rdtypes.ANY.CNAME.CNAME(rdclass, rdtype, dns.name.from_text(cname)))
        return HostsAnswer(qname, rdtype, rdclass, rrset, raise_on_no_answer)

    def getaliases(self, hostname):
        """Return a list of all the aliases of a given cname"""
        aliases = []
        if hostname in self._aliases:
            cannon = self._aliases[hostname]
        else:
            cannon = hostname
        aliases.append(cannon)
        for alias, cname in self._aliases.items():
            if cannon == cname:
                aliases.append(alias)
        aliases.remove(hostname)
        return aliases