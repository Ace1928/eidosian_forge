import time
from . import debug, errors, osutils, revision, trace
class DictParentsProvider:
    """A parents provider for Graph objects."""

    def __init__(self, ancestry):
        self.ancestry = ancestry

    def __repr__(self):
        return 'DictParentsProvider(%r)' % self.ancestry

    def get_parent_map(self, keys):
        """See StackedParentsProvider.get_parent_map"""
        ancestry = self.ancestry
        return {k: ancestry[k] for k in keys if k in ancestry}