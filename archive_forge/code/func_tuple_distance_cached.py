from .data_dicts import LANGUAGE_DISTANCES
from typing import Dict, Tuple
def tuple_distance_cached(desired: TagTriple, supported: TagTriple) -> int:
    """
    Takes in triples of (language, script, territory), which can be derived by
    'maximizing' a language tag. Returns a number from 0 to 135 indicating the
    'distance' between these for the purposes of language matching.
    """
    if supported == desired:
        return 0
    if (desired, supported) in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[desired, supported]
    else:
        result = _tuple_distance(desired, supported)
        _DISTANCE_CACHE[desired, supported] = result
        return result