from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class SharedInstrumentedParentsProvider:

    def __init__(self, parents_provider, calls, info):
        self.calls = calls
        self.info = info
        self._real_parents_provider = parents_provider
        get_cached = getattr(parents_provider, 'get_cached_parent_map', None)
        if get_cached is not None:
            self.get_cached_parent_map = self._get_cached_parent_map

    def get_parent_map(self, nodes):
        self.calls.append((self.info, sorted(nodes)))
        return self._real_parents_provider.get_parent_map(nodes)

    def _get_cached_parent_map(self, nodes):
        self.calls.append((self.info, 'cached', sorted(nodes)))
        return self._real_parents_provider.get_cached_parent_map(nodes)