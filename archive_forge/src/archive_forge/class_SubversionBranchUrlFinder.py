from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
class SubversionBranchUrlFinder:

    def __init__(self):
        self._roots = defaultdict(set)

    def find_root(self, uuid, url):
        for root in self._roots[uuid]:
            if url.startswith(root):
                return root
        try:
            from subvertpy.ra import RemoteAccess
        except ModuleNotFoundError:
            return None
        c = RemoteAccess(url)
        root = c.get_repos_root()
        self._roots[uuid].add(root)
        return root

    def find_branch_path(self, uuid, url):
        root = self.find_root(uuid, url)
        if root is None:
            return None
        assert url.startswith(root)
        return url[len(root):].strip('/')