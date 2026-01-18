from kivy.config import Config
from kivy.vector import Vector
import time

    InputPostprocRetainTouch is a post-processor to delay the 'up' event of a
    touch, to reuse it under certains conditions. This module is designed to
    prevent lost finger touches on some hardware/setups.

    Retain touch can be configured in the Kivy config file::

        [postproc]
            retain_time = 100
            retain_distance = 50

    The distance parameter is in the range 0-1000 and time is in milliseconds.
    