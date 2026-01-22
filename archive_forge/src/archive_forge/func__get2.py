from .data_dicts import LANGUAGE_DISTANCES
from typing import Dict, Tuple
def _get2(dictionary: dict, key1: str, key2: str, default):
    return dictionary.get(key1, {}).get(key2, default)