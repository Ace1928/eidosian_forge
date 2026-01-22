import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _scroll_callback(self, window, x_offset, y_offset):
    with self._gui_lock:
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)