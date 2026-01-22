import os
import io
from .._utils import set_module
class _FileOpeners:
    """
    Container for different methods to open (un-)compressed files.

    `_FileOpeners` contains a dictionary that holds one method for each
    supported file format. Attribute lookup is implemented in such a way
    that an instance of `_FileOpeners` itself can be indexed with the keys
    of that dictionary. Currently uncompressed files as well as files
    compressed with ``gzip``, ``bz2`` or ``xz`` compression are supported.

    Notes
    -----
    `_file_openers`, an instance of `_FileOpeners`, is made available for
    use in the `_datasource` module.

    Examples
    --------
    >>> import gzip
    >>> np.lib._datasource._file_openers.keys()
    [None, '.bz2', '.gz', '.xz', '.lzma']
    >>> np.lib._datasource._file_openers['.gz'] is gzip.open
    True

    """

    def __init__(self):
        self._loaded = False
        self._file_openers = {None: io.open}

    def _load(self):
        if self._loaded:
            return
        try:
            import bz2
            self._file_openers['.bz2'] = bz2.open
        except ImportError:
            pass
        try:
            import gzip
            self._file_openers['.gz'] = gzip.open
        except ImportError:
            pass
        try:
            import lzma
            self._file_openers['.xz'] = lzma.open
            self._file_openers['.lzma'] = lzma.open
        except (ImportError, AttributeError):
            pass
        self._loaded = True

    def keys(self):
        """
        Return the keys of currently supported file openers.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list
            The keys are None for uncompressed files and the file extension
            strings (i.e. ``'.gz'``, ``'.xz'``) for supported compression
            methods.

        """
        self._load()
        return list(self._file_openers.keys())

    def __getitem__(self, key):
        self._load()
        return self._file_openers[key]