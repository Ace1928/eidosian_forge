from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
class ObjectStoreParentsProvider:

    def __init__(self, store):
        self._store = store

    def get_parent_map(self, shas):
        ret = {}
        for sha in shas:
            if sha is None:
                parents = []
            else:
                try:
                    parents = self._store[sha].parents
                except KeyError:
                    parents = None
            ret[sha] = parents
        return ret