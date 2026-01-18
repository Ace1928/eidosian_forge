import functools
import pathlib
from contextlib import suppress
from types import SimpleNamespace
from .. import readers, _adapters
def wrap_spec(package):
    """
    Override _adapters.wrap_spec to use TraversableResourcesLoader
    from above. Ensures that future behavior is always available on older
    Pythons.
    """
    return _adapters.SpecLoaderAdapter(package.__spec__, TraversableResourcesLoader)