import sys
import requests
def _copy_from(self, other):
    for key in other:
        val = other.getlist(key)
        if isinstance(val, list):
            val = list(val)
        self._container[key.lower()] = [key] + val