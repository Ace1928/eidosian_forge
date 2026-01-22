from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
class PyDecoder(PyCodec):
    """
    Python implementation of a format decoder. Override this class and
    add the decoding logic in the :meth:`decode` method.

    See :ref:`Writing Your Own File Codec in Python<file-codecs-py>`
    """
    _pulls_fd = False

    @property
    def pulls_fd(self):
        return self._pulls_fd

    def decode(self, buffer):
        """
        Override to perform the decoding process.

        :param buffer: A bytes object with the data to be decoded.
        :returns: A tuple of ``(bytes consumed, errcode)``.
            If finished with decoding return -1 for the bytes consumed.
            Err codes are from :data:`.ImageFile.ERRORS`.
        """
        msg = 'unavailable in base decoder'
        raise NotImplementedError(msg)

    def set_as_raw(self, data, rawmode=None):
        """
        Convenience method to set the internal image from a stream of raw data

        :param data: Bytes to be set
        :param rawmode: The rawmode to be used for the decoder.
            If not specified, it will default to the mode of the image
        :returns: None
        """
        if not rawmode:
            rawmode = self.mode
        d = Image._getdecoder(self.mode, 'raw', rawmode)
        d.setimage(self.im, self.state.extents())
        s = d.decode(data)
        if s[0] >= 0:
            msg = 'not enough image data'
            raise ValueError(msg)
        if s[1] != 0:
            msg = 'cannot decode image data'
            raise ValueError(msg)