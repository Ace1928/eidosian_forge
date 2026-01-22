import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _init_camera(self):
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cam.fixedcamid = -1
    for i in range(3):
        self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
    self.cam.distance = self.model.stat.extent