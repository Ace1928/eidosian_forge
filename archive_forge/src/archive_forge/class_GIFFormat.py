import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
class GIFFormat(PillowFormat):
    """See :mod:`imageio.plugins.pillow_legacy`"""
    _modes = 'iI'
    _description = 'Static and animated gif (Pillow)'

    class Writer(PillowFormat.Writer):

        def _open(self, loop=0, duration=None, fps=10, palettesize=256, quantizer=0, subrectangles=False):
            from PIL import __version__ as pillow_version
            major, minor, patch = tuple((int(x) for x in pillow_version.split('.')))
            if major == 10 and minor >= 1:
                raise ImportError(f"Pillow v{pillow_version} is not supported by ImageIO's legacy pillow plugin when writing GIFs. Consider switching to the new plugin or downgrading to `pillow<10.1.0`.")
            palettesize = int(palettesize)
            if palettesize < 2 or palettesize > 256:
                raise ValueError('GIF quantize param must be 2..256')
            if palettesize not in [2, 4, 8, 16, 32, 64, 128, 256]:
                palettesize = 2 ** int(np.log2(128) + 0.999)
                logger.warning('Warning: palettesize (%r) modified to a factor of two between 2-256.' % palettesize)
            if duration is None:
                self._duration = 1.0 / float(fps)
            elif isinstance(duration, (list, tuple)):
                self._duration = [float(d) for d in duration]
            else:
                self._duration = float(duration)
            loop = float(loop)
            if loop <= 0 or loop == float('inf'):
                loop = 0
            loop = int(loop)
            subrectangles = bool(subrectangles)
            self._dispose = 1 if subrectangles else 2
            fp = self.request.get_file()
            self._writer = GifWriter(fp, subrectangles, loop, quantizer, int(palettesize))

        def _close(self):
            self._writer.close()

        def _append_data(self, im, meta):
            im = image_as_uint(im, bitdepth=8)
            if im.ndim == 3 and im.shape[-1] == 1:
                im = im[:, :, 0]
            duration = self._duration
            if isinstance(duration, list):
                duration = duration[min(len(duration) - 1, self._writer._count)]
            dispose = self._dispose
            self._writer.add_image(im, duration, dispose)
            return