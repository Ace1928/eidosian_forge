from kivy.properties import ObjectProperty
from kivy.event import EventDispatcher
from collections import defaultdict
def _clean_cache():
    """Trims _cached_views cache to half the size of `_max_cache_size`.
    """
    max_size = _max_cache_size // 2 // len(_cached_views)
    global _cache_count
    for cls, instances in _cached_views.items():
        _cache_count -= max(0, len(instances) - max_size)
        del instances[max_size:]