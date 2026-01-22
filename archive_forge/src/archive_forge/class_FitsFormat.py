from ..core import Format
class FitsFormat(Format):
    """See :mod:`imageio.plugins.fits`"""

    def _can_read(self, request):
        return request.extension in self.extensions

    def _can_write(self, request):
        return False

    class Reader(Format.Reader):

        def _open(self, cache=False, **kwargs):
            if not _fits:
                load_lib()
            hdulist = _fits.open(self.request.get_file(), cache=cache, **kwargs)
            self._index = []
            allowed_hdu_types = (_fits.ImageHDU, _fits.PrimaryHDU, _fits.CompImageHDU)
            for n, hdu in zip(range(len(hdulist)), hdulist):
                if isinstance(hdu, allowed_hdu_types):
                    if hdu.size > 0:
                        self._index.append(n)
            self._hdulist = hdulist

        def _close(self):
            self._hdulist.close()

        def _get_length(self):
            return len(self._index)

        def _get_data(self, index):
            if index < 0 or index >= len(self._index):
                raise IndexError('Index out of range while reading from fits')
            im = self._hdulist[self._index[index]].data
            return (im, {})

        def _get_meta_data(self, index):
            raise RuntimeError('The fits format does not support meta data.')