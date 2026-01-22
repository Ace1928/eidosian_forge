from statsmodels.tools.sm_exceptions import CacheWriteWarning
from statsmodels.compat.pandas import cache_readonly as PandasCacheReadonly
import warnings
class cache_writable(_cache_readonly):
    """
    Decorator for CachedWritableAttribute
    """

    def __call__(self, func):
        return CachedWritableAttribute(func, cachename=self.cachename)