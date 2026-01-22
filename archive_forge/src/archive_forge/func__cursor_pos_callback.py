import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _cursor_pos_callback(self, window, xpos, ypos):
    if not (self._button_left_pressed or self._button_right_pressed):
        return
    mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    if self._button_right_pressed:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif self._button_left_pressed:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
    dx = int(self._scale * xpos) - self._last_mouse_x
    dy = int(self._scale * ypos) - self._last_mouse_y
    width, height = glfw.get_framebuffer_size(window)
    with self._gui_lock:
        mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height, self.scn, self.cam)
    self._last_mouse_x = int(self._scale * xpos)
    self._last_mouse_y = int(self._scale * ypos)