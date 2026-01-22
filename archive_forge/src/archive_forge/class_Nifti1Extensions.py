from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
class Nifti1Extensions(list):
    """Simple extension collection, implemented as a list-subclass."""

    def count(self, ecode):
        """Returns the number of extensions matching a given *ecode*.

        Parameters
        ----------
        code : int | str
            The ecode can be specified either literal or as numerical value.
        """
        count = 0
        code = extension_codes.code[ecode]
        for e in self:
            if e.get_code() == code:
                count += 1
        return count

    def get_codes(self):
        """Return a list of the extension code of all available extensions"""
        return [e.get_code() for e in self]

    def get_sizeondisk(self):
        """Return the size of the complete header extensions in the NIfTI file."""
        return np.sum([e.get_sizeondisk() for e in self])

    def __repr__(self):
        return 'Nifti1Extensions(%s)' % ', '.join((str(e) for e in self))

    def write_to(self, fileobj, byteswap):
        """Write header extensions to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method
        byteswap : boolean
          Flag if byteswapping the data is required.

        Returns
        -------
        None
        """
        for e in self:
            e.write_to(fileobj, byteswap)

    @classmethod
    def from_fileobj(klass, fileobj, size, byteswap):
        """Read header extensions from a fileobj

        Parameters
        ----------
        fileobj : file-like object
            We begin reading the extensions at the current file position
        size : int
            Number of bytes to read. If negative, fileobj will be read till its
            end.
        byteswap : boolean
            Flag if byteswapping the read data is required.

        Returns
        -------
        An extension list. This list might be empty in case not extensions
        were present in fileobj.
        """
        extensions = klass()
        while size >= 16 or size < 0:
            ext_def = fileobj.read(8)
            if not len(ext_def) and size < 0:
                break
            if not len(ext_def) == 8:
                raise HeaderDataError('failed to read extension header')
            ext_def = np.frombuffer(ext_def, dtype=np.int32)
            if byteswap:
                ext_def = ext_def.byteswap()
            ecode = ext_def[1]
            esize = ext_def[0]
            if esize % 16:
                warnings.warn('Extension size is not a multiple of 16 bytes; Assuming size is correct and hoping for the best', UserWarning)
            evalue = fileobj.read(int(esize - 8))
            if not len(evalue) == esize - 8:
                raise HeaderDataError('failed to read extension content')
            size -= esize
            evalue = evalue.rstrip(b'\x00')
            try:
                ext = extension_codes.handler[ecode](ecode, evalue)
            except KeyError:
                ext = Nifti1Extension(ecode, evalue)
            extensions.append(ext)
        return extensions