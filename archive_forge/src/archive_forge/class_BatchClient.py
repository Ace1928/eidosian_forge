import time
import email
import socket
import logging
import functools
import collections
import pyzor.digest
import pyzor.account
import pyzor.message
import pyzor.hacks.py26
class BatchClient(Client):
    """Like the normal Client but with support for batching reports."""

    def __init__(self, accounts=None, timeout=None, spec=None, batch_size=10):
        Client.__init__(self, accounts=accounts, timeout=timeout, spec=spec)
        self.batch_size = batch_size
        self.r_requests = {}
        self.w_requests = {}
        self.flush()

    def report(self, digest, address=('public.pyzor.org', 24441)):
        self._add_digest(digest, address, self.r_requests)

    def whitelist(self, digest, address=('public.pyzor.org', 24441)):
        self._add_digest(digest, address, self.w_requests)

    def _add_digest(self, digest, address, requests):
        address = (address[0], int(address[1]))
        msg = requests[address]
        msg.add_digest(digest)
        if msg.digest_count >= self.batch_size:
            try:
                return self.send(msg, address)
            finally:
                del requests[address]

    def flush(self):
        """Deleting any saved digest reports."""
        self.r_requests = collections.defaultdict(functools.partial(pyzor.message.ReportRequest, spec=self.spec))
        self.w_requests = collections.defaultdict(functools.partial(pyzor.message.WhitelistRequest, spec=self.spec))

    def force(self):
        """Force send any remaining reports."""
        for address, msg in self.r_requests.iteritems():
            try:
                self.send(msg, address)
            except:
                continue
        for address, msg in self.w_requests.iteritems():
            try:
                self.send(msg, address)
            except:
                continue

    def __del__(self):
        self.force()