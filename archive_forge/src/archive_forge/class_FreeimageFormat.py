import numpy as np
from ..core import Format, image_as_uint
from ..core.request import RETURN_BYTES
from ._freeimage import FNAME_PER_PLATFORM, IO_FLAGS, download, fi  # noqa
class FreeimageFormat(Format):
    """See :mod:`imageio.plugins.freeimage`"""
    _modes = 'i'

    def __init__(self, name, description, extensions=None, modes=None, *, fif=None):
        super().__init__(name, description, extensions=extensions, modes=modes)
        self._fif = fif

    @property
    def fif(self):
        return self._fif

    def _can_read(self, request):
        if fi.has_lib():
            if not hasattr(request, '_fif'):
                try:
                    request._fif = fi.getFIF(request.filename, 'r', request.firstbytes)
                except Exception:
                    request._fif = -1
            if request._fif == self.fif:
                return True
            elif request._fif == 7 and self.fif == 14:
                return True

    def _can_write(self, request):
        if fi.has_lib():
            if not hasattr(request, '_fif'):
                try:
                    request._fif = fi.getFIF(request.filename, 'w')
                except ValueError:
                    if request.raw_uri == RETURN_BYTES:
                        request._fif = self.fif
                    else:
                        request._fif = -1
            if request._fif is self.fif:
                return True

    class Reader(Format.Reader):

        def _get_length(self):
            return 1

        def _open(self, flags=0):
            self._bm = fi.create_bitmap(self.request.filename, self.format.fif, flags)
            self._bm.load_from_filename(self.request.get_local_filename())

        def _close(self):
            self._bm.close()

        def _get_data(self, index):
            if index != 0:
                raise IndexError('This format only supports singleton images.')
            return (self._bm.get_image_data(), self._bm.get_meta_data())

        def _get_meta_data(self, index):
            if not (index is None or index == 0):
                raise IndexError()
            return self._bm.get_meta_data()

    class Writer(Format.Writer):

        def _open(self, flags=0):
            self._flags = flags
            self._bm = None
            self._is_set = False
            self._meta = {}

        def _close(self):
            self._bm.set_meta_data(self._meta)
            self._bm.save_to_filename(self.request.get_local_filename())
            self._bm.close()

        def _append_data(self, im, meta):
            if not self._is_set:
                self._is_set = True
            else:
                raise RuntimeError('Singleton image; can only append image data once.')
            if im.ndim == 3 and im.shape[-1] == 1:
                im = im[:, :, 0]
            if self._bm is None:
                self._bm = fi.create_bitmap(self.request.filename, self.format.fif, self._flags)
                self._bm.allocate(im)
            self._bm.set_image_data(im)
            self._meta = meta

        def _set_meta_data(self, meta):
            self._meta = meta