from __future__ import annotations
import io
import typing as ty
from copy import deepcopy
from urllib import request
from ._compression import COMPRESSION_ERRORS
from .fileholders import FileHolder, FileMap
from .filename_parser import TypesFilenamesError, _stringify_path, splitext_addext, types_filenames
from .openers import ImageOpener
class SerializableImage(FileBasedImage):
    """
    Abstract image class for (de)serializing images to/from byte streams/strings.

    The class doesn't define any image properties.

    It has:

    methods:

       * to_bytes() - serialize image to byte string

    classmethods:

       * from_bytes(bytestring) - make instance by deserializing a byte string
       * from_url(url) - make instance by fetching and deserializing a URL

    Loading from byte strings should provide round-trip equivalence:

    .. code:: python

        img_a = klass.from_bytes(bstr)
        img_b = klass.from_bytes(img_a.to_bytes())

        np.allclose(img_a.get_fdata(), img_b.get_fdata())
        np.allclose(img_a.affine, img_b.affine)

    Further, for images that are single files on disk, the following methods of loading
    the image must be equivalent:

    .. code:: python

        img = klass.from_filename(fname)

        with open(fname, 'rb') as fobj:
            img = klass.from_bytes(fobj.read())

    And the following methods of saving a file must be equivalent:

    .. code:: python

        img.to_filename(fname)

        with open(fname, 'wb') as fobj:
            fobj.write(img.to_bytes())

    Images that consist of separate header and data files (e.g., Analyze
    images) currently do not support this interface.
    For multi-file images, ``to_bytes()`` and ``from_bytes()`` must be
    overridden, and any encoding details should be documented.
    """

    @classmethod
    def _filemap_from_iobase(klass, io_obj: io.IOBase) -> FileMap:
        """For single-file image types, make a file map with the correct key"""
        if len(klass.files_types) > 1:
            raise NotImplementedError('(de)serialization is undefined for multi-file images')
        return klass.make_file_map({klass.files_types[0][0]: io_obj})

    @classmethod
    def from_stream(klass: type[StreamImgT], io_obj: io.IOBase) -> StreamImgT:
        """Load image from readable IO stream

        Convert to BytesIO to enable seeking, if input stream is not seekable

        Parameters
        ----------
        io_obj : IOBase object
            Readable stream
        """
        if not io_obj.seekable():
            io_obj = io.BytesIO(io_obj.read())
        return klass.from_file_map(klass._filemap_from_iobase(io_obj))

    def to_stream(self, io_obj: io.IOBase, **kwargs) -> None:
        """Save image to writable IO stream

        Parameters
        ----------
        io_obj : IOBase object
            Writable stream
        \\*\\*kwargs : keyword arguments
            Keyword arguments that may be passed to ``img.to_file_map()``
        """
        self.to_file_map(self._filemap_from_iobase(io_obj), **kwargs)

    @classmethod
    def from_bytes(klass: type[StreamImgT], bytestring: bytes) -> StreamImgT:
        """Construct image from a byte string

        Class method

        Parameters
        ----------
        bytestring : bytes
            Byte string containing the on-disk representation of an image
        """
        return klass.from_stream(io.BytesIO(bytestring))

    def to_bytes(self, **kwargs) -> bytes:
        """Return a ``bytes`` object with the contents of the file that would
        be written if the image were saved.

        Parameters
        ----------
        \\*\\*kwargs : keyword arguments
            Keyword arguments that may be passed to ``img.to_file_map()``

        Returns
        -------
        bytes
            Serialized image
        """
        bio = io.BytesIO()
        self.to_stream(bio, **kwargs)
        return bio.getvalue()

    @classmethod
    def from_url(klass: type[StreamImgT], url: str | request.Request, timeout: float=5) -> StreamImgT:
        """Retrieve and load an image from a URL

        Class method

        Parameters
        ----------
        url : str or urllib.request.Request object
            URL of file to retrieve
        timeout : float, optional
            Time (in seconds) to wait for a response
        """
        response = request.urlopen(url, timeout=timeout)
        return klass.from_stream(response)