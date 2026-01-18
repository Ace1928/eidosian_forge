import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
Imports and caches pre-defined API.

    Warns if necessary.

    This method is a replacement for __getattr__(). It will be added into the
    extended python module as a callback to reduce API overhead. Instead of
    relying on implicit AttributeError handling, this added callback function
    will
    be called explicitly from the extended C API if the default attribute lookup
    fails.
    