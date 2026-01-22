import abc
import os
import bz2
import gzip
import lzma
import shutil
from zipfile import ZipFile
from tarfile import TarFile
from .utils import get_logger
class Decompress:
    """
    Processor that decompress a file and returns the decompressed version.

    Use with :meth:`pooch.Pooch.fetch` or :func:`pooch.retrieve` to decompress
    a downloaded data file so that it can be easily opened. Useful for data
    files that take a long time to decompress (exchanging disk space for
    speed).

    Supported decompression methods are LZMA (``.xz``), bzip2 (``.bz2``), and
    gzip (``.gz``).

    File names with the standard extensions (see above) can use
    ``method="auto"`` to automatically determine the compression method. This
    can be overwritten by setting the *method* argument.

    .. note::

        To unpack zip and tar archives with one or more files, use
        :class:`pooch.Unzip` and :class:`pooch.Untar` instead.

    The output file is ``{fname}.decomp`` by default but it can be changed by
    setting the ``name`` parameter.

    .. warning::

        Passing in ``name`` can cause existing data to be lost! For example, if
        a file already exists with the specified name it will be overwritten
        with the new decompressed file content. **Use this option with
        caution.**

    Parameters
    ----------
    method : str
        Name of the compression method. Can be "auto", "lzma", "xz", "bzip2",
        or "gzip".
    name : None or str
        Defines the decompressed file name. The file name will be
        ``{fname}.decomp`` if ``None`` (default) or the given name otherwise.
        Note that the name should **not** include the full (or relative) path,
        it should be just the file name itself.

    """
    modules = {'auto': None, 'lzma': lzma, 'xz': lzma, 'gzip': gzip, 'bzip2': bz2}
    extensions = {'.xz': 'lzma', '.gz': 'gzip', '.bz2': 'bzip2'}

    def __init__(self, method='auto', name=None):
        self.method = method
        self.name = name

    def __call__(self, fname, action, pooch):
        """
        Decompress the given file.

        The output file will be either ``{fname}.decomp`` or the given *name*
        class attribute.

        Parameters
        ----------
        fname : str
            Full path of the compressed file in local storage.
        action : str
            Indicates what action was taken by :meth:`pooch.Pooch.fetch` or
            :func:`pooch.retrieve`:

            - ``"download"``: File didn't exist locally and was downloaded
            - ``"update"``: Local file was outdated and was re-download
            - ``"fetch"``: File exists and is updated so it wasn't downloaded

        pooch : :class:`pooch.Pooch`
            The instance of :class:`pooch.Pooch` that is calling this.

        Returns
        -------
        fname : str
            The full path to the decompressed file.
        """
        if self.name is None:
            decompressed = fname + '.decomp'
        else:
            decompressed = os.path.join(os.path.dirname(fname), self.name)
        if action in ('update', 'download') or not os.path.exists(decompressed):
            get_logger().info("Decompressing '%s' to '%s' using method '%s'.", fname, decompressed, self.method)
            module = self._compression_module(fname)
            with open(decompressed, 'w+b') as output:
                with module.open(fname) as compressed:
                    shutil.copyfileobj(compressed, output)
        return decompressed

    def _compression_module(self, fname):
        """
        Get the Python module compatible with fname and the chosen method.

        If the *method* attribute is "auto", will select a method based on the
        extension. If no recognized extension is in the file name, will raise a
        ValueError.
        """
        error_archives = 'To unpack zip/tar archives, use pooch.Unzip/Untar instead.'
        if self.method not in self.modules:
            message = f"Invalid compression method '{self.method}'. Must be one of '{list(self.modules.keys())}'."
            if self.method in {'zip', 'tar'}:
                message = ' '.join([message, error_archives])
            raise ValueError(message)
        if self.method == 'auto':
            ext = os.path.splitext(fname)[-1]
            if ext not in self.extensions:
                message = f"Unrecognized file extension '{ext}'. Must be one of '{list(self.extensions.keys())}'."
                if ext in {'.zip', '.tar'}:
                    message = ' '.join([message, error_archives])
                raise ValueError(message)
            return self.modules[self.extensions[ext]]
        return self.modules[self.method]