import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
class DistributedInvalidationStrategy(region.RegionInvalidationStrategy):

    def __init__(self, region_manager):
        self._region_manager = region_manager

    def invalidate(self, hard=None):
        self._region_manager.invalidate_region()

    def is_invalidated(self, timestamp):
        return False

    def was_hard_invalidated(self):
        return False

    def is_hard_invalidated(self, timestamp):
        return False

    def was_soft_invalidated(self):
        return False

    def is_soft_invalidated(self, timestamp):
        return False