import numpy as np
from scipy.special import factorial
class Sterling1:
    """Stirling numbers of the first kind
    """

    def __init__(self):
        self._cache = {}

    def __call__(self, n, k):
        key = str(n) + ',' + str(k)
        if key in self._cache.keys():
            return self._cache[key]
        if n == k == 0:
            return 1
        if n > 0 and k == 0:
            return 0
        if k > n:
            return 0
        result = sterling1(n - 1, k - 1) + (n - 1) * sterling1(n - 1, k)
        self._cache[key] = result
        return result

    def clear_cache(self):
        """clear cache of Sterling numbers
        """
        self._cache = {}