import numpy as np
from ..core import Format, image_as_uint
from ..core.request import RETURN_BYTES
from ._freeimage import FNAME_PER_PLATFORM, IO_FLAGS, download, fi  # noqa
class FreeimageJpegFormat(FreeimageFormat):
    """A JPEG format based on the Freeimage library.

    This format supports grayscale and RGB images.

    The freeimage plugin requires a `freeimage` binary. If this binary
    not available on the system, it can be downloaded manually from
    <https://github.com/imageio/imageio-binaries> by either

    - the command line script ``imageio_download_bin freeimage``
    - the Python method ``imageio.plugins.freeimage.download()``

    Parameters for reading
    ----------------------
    exifrotate : bool
        Automatically rotate the image according to the exif flag.
        Default True. If 2 is given, do the rotation in Python instead
        of freeimage.
    quickread : bool
        Read the image more quickly, at the expense of quality.
        Default False.

    Parameters for saving
    ---------------------
    quality : scalar
        The compression factor of the saved image (1..100), higher
        numbers result in higher quality but larger file size. Default 75.
    progressive : bool
        Save as a progressive JPEG file (e.g. for images on the web).
        Default False.
    optimize : bool
        On saving, compute optimal Huffman coding tables (can reduce a
        few percent of file size). Default False.
    baseline : bool
        Save basic JPEG, without metadata or any markers. Default False.

    """

    class Reader(FreeimageFormat.Reader):

        def _open(self, flags=0, exifrotate=True, quickread=False):
            flags = int(flags)
            if exifrotate and exifrotate != 2:
                flags |= IO_FLAGS.JPEG_EXIFROTATE
            if not quickread:
                flags |= IO_FLAGS.JPEG_ACCURATE
            return FreeimageFormat.Reader._open(self, flags)

        def _get_data(self, index):
            im, meta = FreeimageFormat.Reader._get_data(self, index)
            im = self._rotate(im, meta)
            return (im, meta)

        def _rotate(self, im, meta):
            """Use Orientation information from EXIF meta data to
            orient the image correctly. Freeimage is also supposed to
            support that, and I am pretty sure it once did, but now it
            does not, so let's just do it in Python.
            Edit: and now it works again, just leave in place as a fallback.
            """
            if self.request.kwargs.get('exifrotate', None) == 2:
                try:
                    ori = meta['EXIF_MAIN']['Orientation']
                except KeyError:
                    pass
                else:
                    if ori in [1, 2]:
                        pass
                    if ori in [3, 4]:
                        im = np.rot90(im, 2)
                    if ori in [5, 6]:
                        im = np.rot90(im, 3)
                    if ori in [7, 8]:
                        im = np.rot90(im)
                    if ori in [2, 4, 5, 7]:
                        im = np.fliplr(im)
            return im

    class Writer(FreeimageFormat.Writer):

        def _open(self, flags=0, quality=75, progressive=False, optimize=False, baseline=False):
            quality = int(quality)
            if quality < 1 or quality > 100:
                raise ValueError('JPEG quality should be between 1 and 100.')
            flags = int(flags)
            flags |= quality
            if progressive:
                flags |= IO_FLAGS.JPEG_PROGRESSIVE
            if optimize:
                flags |= IO_FLAGS.JPEG_OPTIMIZE
            if baseline:
                flags |= IO_FLAGS.JPEG_BASELINE
            return FreeimageFormat.Writer._open(self, flags)

        def _append_data(self, im, meta):
            if im.ndim == 3 and im.shape[-1] == 4:
                raise IOError('JPEG does not support alpha channel.')
            im = image_as_uint(im, bitdepth=8)
            return FreeimageFormat.Writer._append_data(self, im, meta)