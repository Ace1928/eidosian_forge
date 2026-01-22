import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
class PNGFormat(PillowFormat):
    """See :mod:`imageio.plugins.pillow_legacy`"""

    class Reader(PillowFormat.Reader):

        def _open(self, pilmode=None, as_gray=False, ignoregamma=True):
            return PillowFormat.Reader._open(self, pilmode=pilmode, as_gray=as_gray)

        def _get_data(self, index):
            im, info = PillowFormat.Reader._get_data(self, index)
            if not self.request.kwargs.get('ignoregamma', True):
                try:
                    gamma = float(info['gamma'])
                except (KeyError, ValueError):
                    pass
                else:
                    scale = float(65536 if im.dtype == np.uint16 else 255)
                    gain = 1.0
                    im[:] = (im / scale) ** gamma * scale * gain + 0.4999
            return (im, info)

    class Writer(PillowFormat.Writer):

        def _open(self, compression=None, quantize=None, interlaced=False, **kwargs):
            kwargs['compress_level'] = kwargs.get('compress_level', 9)
            if compression is not None:
                if compression < 0 or compression > 9:
                    raise ValueError('Invalid PNG compression level: %r' % compression)
                kwargs['compress_level'] = compression
            if quantize is not None:
                for bits in range(1, 9):
                    if 2 ** bits == quantize:
                        break
                else:
                    raise ValueError('PNG quantize must be power of two, not %r' % quantize)
                kwargs['bits'] = bits
            if interlaced:
                logger.warning('PIL PNG writer cannot produce interlaced images.')
            ok_keys = ('optimize', 'transparency', 'dpi', 'pnginfo', 'bits', 'compress_level', 'icc_profile', 'dictionary', 'prefer_uint8')
            for key in kwargs:
                if key not in ok_keys:
                    raise TypeError('Invalid arg for PNG writer: %r' % key)
            PillowFormat.Writer._open(self)
            self._meta.update(kwargs)

        def _append_data(self, im, meta):
            if str(im.dtype) == 'uint16' and (im.ndim == 2 or im.shape[-1] == 1):
                im = image_as_uint(im, bitdepth=16)
            else:
                im = image_as_uint(im, bitdepth=8)
            PillowFormat.Writer._append_data(self, im, meta)