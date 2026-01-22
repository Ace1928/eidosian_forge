from typing import Dict, Tuple, Callable
def _create_get_from_cache(number: int) -> Callable[[str, str, CacheValuesCallback], str]:

    def _get_from_cache(module_name: str, name: str, get_cache_values: CacheValuesCallback) -> str:
        try:
            return _cache[module_name][name][number]
        except KeyError:
            v = get_cache_values()
            save_entry(module_name, name, v)
            return v[number]
    return _get_from_cache