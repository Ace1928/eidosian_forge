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
class NXDOMAIN(dns.exception.DNSException):
    """The DNS query name does not exist."""
    supp_kwargs = {'qnames', 'responses'}
    fmt = None

    def _check_kwargs(self, qnames, responses=None):
        if not isinstance(qnames, (list, tuple, set)):
            raise AttributeError('qnames must be a list, tuple or set')
        if len(qnames) == 0:
            raise AttributeError('qnames must contain at least one element')
        if responses is None:
            responses = {}
        elif not isinstance(responses, dict):
            raise AttributeError('responses must be a dict(qname=response)')
        kwargs = dict(qnames=qnames, responses=responses)
        return kwargs

    def __str__(self):
        if 'qnames' not in self.kwargs:
            return super(NXDOMAIN, self).__str__()
        qnames = self.kwargs['qnames']
        if len(qnames) > 1:
            msg = 'None of DNS query names exist'
        else:
            msg = 'The DNS query name does not exist'
        qnames = ', '.join(map(str, qnames))
        return '{}: {}'.format(msg, qnames)

    def canonical_name(self):
        if not 'qnames' in self.kwargs:
            raise TypeError('parametrized exception required')
        IN = dns.rdataclass.IN
        CNAME = dns.rdatatype.CNAME
        cname = None
        for qname in self.kwargs['qnames']:
            response = self.kwargs['responses'][qname]
            for answer in response.answer:
                if answer.rdtype != CNAME or answer.rdclass != IN:
                    continue
                cname = answer.items[0].target.to_text()
            if cname is not None:
                return dns.name.from_text(cname)
        return self.kwargs['qnames'][0]
    canonical_name = property(canonical_name, doc='Return the unresolved canonical name.')

    def __add__(self, e_nx):
        """Augment by results from another NXDOMAIN exception."""
        qnames0 = list(self.kwargs.get('qnames', []))
        responses0 = dict(self.kwargs.get('responses', {}))
        responses1 = e_nx.kwargs.get('responses', {})
        for qname1 in e_nx.kwargs.get('qnames', []):
            if qname1 not in qnames0:
                qnames0.append(qname1)
            if qname1 in responses1:
                responses0[qname1] = responses1[qname1]
        return NXDOMAIN(qnames=qnames0, responses=responses0)

    def qnames(self):
        """All of the names that were tried.

        Returns a list of ``dns.name.Name``.
        """
        return self.kwargs['qnames']

    def responses(self):
        """A map from queried names to their NXDOMAIN responses.

        Returns a dict mapping a ``dns.name.Name`` to a
        ``dns.message.Message``.
        """
        return self.kwargs['responses']

    def response(self, qname):
        """The response for query *qname*.

        Returns a ``dns.message.Message``.
        """
        return self.kwargs['responses'][qname]