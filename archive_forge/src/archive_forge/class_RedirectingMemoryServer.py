import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
class RedirectingMemoryServer(memory.MemoryServer):

    def start_server(self):
        self._dirs = {'/': None}
        self._files = {}
        self._locks = {}
        self._scheme = 'redirecting-memory+%s:///' % id(self)
        transport.register_transport(self._scheme, self._memory_factory)

    def _memory_factory(self, url):
        result = RedirectingMemoryTransport(url)
        result._dirs = self._dirs
        result._files = self._files
        result._locks = self._locks
        return result

    def stop_server(self):
        transport.unregister_transport(self._scheme, self._memory_factory)