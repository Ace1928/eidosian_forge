import threading
import numpy as np
from ..core import Format
class ScreenGrabFormat(BaseGrabFormat):
    """The ScreenGrabFormat provided a means to grab screenshots using
    the uri of "<screen>".

    This functionality is provided via Pillow. Note that "<screen>" is
    only supported on Windows and OS X.

    Parameters for reading
    ----------------------
    No parameters.
    """

    def _can_read(self, request):
        if request.filename != '<screen>':
            return False
        return bool(self._init_pillow())

    def _get_data(self, index):
        ImageGrab = self._init_pillow()
        assert ImageGrab
        pil_im = ImageGrab.grab()
        assert pil_im is not None
        im = np.asarray(pil_im)
        return (im, {})