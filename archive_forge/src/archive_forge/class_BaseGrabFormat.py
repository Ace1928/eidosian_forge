import threading
import numpy as np
from ..core import Format
class BaseGrabFormat(Format):
    """Base format for grab formats."""
    _pillow_imported = False
    _ImageGrab = None

    def __init__(self, *args, **kwargs):
        super(BaseGrabFormat, self).__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def _can_write(self, request):
        return False

    def _init_pillow(self):
        with self._lock:
            if not self._pillow_imported:
                self._pillow_imported = True
                import PIL
                if not hasattr(PIL, '__version__'):
                    raise ImportError('Imageio Pillow requires Pillow, not PIL!')
                try:
                    from PIL import ImageGrab
                except ImportError:
                    return None
                self._ImageGrab = ImageGrab
        return self._ImageGrab

    class Reader(Format.Reader):

        def _open(self):
            pass

        def _close(self):
            pass

        def _get_data(self, index):
            return self.format._get_data(index)