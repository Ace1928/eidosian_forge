import functools
import pathlib
from contextlib import suppress
from types import SimpleNamespace
from .. import readers, _adapters
def _block_standard(reader_getter):
    """
    Wrap _adapters.TraversableResourcesLoader.get_resource_reader
    and intercept any standard library readers.
    """

    @functools.wraps(reader_getter)
    def wrapper(*args, **kwargs):
        """
        If the reader is from the standard library, return None to allow
        allow likely newer implementations in this library to take precedence.
        """
        try:
            reader = reader_getter(*args, **kwargs)
        except NotADirectoryError:
            return
        mod_name = reader.__class__.__module__
        if mod_name.startswith('importlib.') and mod_name.endswith('readers'):
            return
        if isinstance(reader, _adapters.CompatibilityFiles) and (reader.spec.loader.__class__.__module__.startswith('zipimport') or reader.spec.loader.__class__.__module__.startswith('_frozen_importlib_external')):
            return
        return reader
    return wrapper