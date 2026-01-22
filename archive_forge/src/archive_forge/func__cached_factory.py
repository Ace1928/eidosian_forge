import inspect
import threading
import types
import gast
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging
def _cached_factory(self, fn, cache_subkey):
    cached_factory = self._cache[fn][cache_subkey]
    logging.log(3, 'Cache hit for %s subkey %s: %s', fn, cache_subkey, cached_factory)
    return cached_factory