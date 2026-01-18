from kivy.config import Config
from kivy.utils import strtotuple

    InputPostprocIgnoreList is a post-processor which removes touches in the
    Ignore list. The Ignore list can be configured in the Kivy config file::

        [postproc]
        # Format: [(xmin, ymin, xmax, ymax), ...]
        ignore = [(0.1, 0.1, 0.15, 0.15)]

    The Ignore list coordinates are in the range 0-1, not in screen pixels.
    